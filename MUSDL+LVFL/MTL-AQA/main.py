import os
import sys

sys.path.append('../')

import torch.nn as nn

from utils import *
from opts import *
from scipy import stats
from tqdm import tqdm
from dataset import VideoDataset
from models.i3d import InceptionI3d
from models.evaluator import Evaluator
from config import get_parser

from transformers import BertTokenizer,BertModel
from moe import MOE
from text import text_prompt

# transformer融合
class Joint_model(nn.Module):
    def __init__(self, ):
        super(Joint_model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('./bert/')
        self.bert = BertModel.from_pretrained("./bert/")
        self.mseloss = nn.MSELoss()
        self.moe = MOE(input_dim=768, num_experts=2, hidden_dim=768)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=768,
                                                            nhead=8,
                                                            dim_feedforward=2048
                                                            )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.gate=nn.Linear(768,1)
        self.layer1 = nn.Linear(1024, 768)
        self.layer2 = nn.Linear(768, 1024)

    def forward(self, clip_feats, text):
        text_input = text_prompt(text)
        text_feature = self.bert(**text_input)[0]
        text_feats = text_feature.mean(1)

        clip_feats = self.layer1(clip_feats.mean(1))

        gate_value = torch.sigmoid(self.gate(text_feats))
        modulated_clip_feats = clip_feats * gate_value

        modulated_feats = self.transformer_encoder(modulated_clip_feats.unsqueeze(1)).squeeze(1)
        joint_feats = self.layer2(modulated_feats)


        text_modulation = self.moe(text_feats)
        loss = self.mseloss(text_modulation, clip_feats)

        return joint_feats, loss




def get_models(args):
    """
    Get the i3d backbone and the evaluator with parameters moved to GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))

    joint_model = Joint_model().cuda()

    evaluator = Evaluator(output_dim=output_dim['MUSDL+LVFL'], model_type='MUSDL+LVFL', num_judges=num_judges).cuda()

    if len(args.gpu.split(',')) > 1:
        i3d = nn.DataParallel(i3d)
        evaluator = nn.DataParallel(evaluator)
        joint_model = nn.DataParallel(joint_model)
    return i3d, joint_model, evaluator


def compute_score( probs, data):
    judge_scores_pred = torch.stack([prob.argmax(dim=-1) * judge_max / (output_dim['MUSDL+LVFL']-1)
                                         for prob in probs], dim=1).sort()[0]  # N, 7

    pred = torch.sum(judge_scores_pred[:, 2:5], dim=1) * data['difficulty'].cuda()
    return pred


def compute_loss(criterion, probs, data):
    loss = sum([criterion(torch.log(probs[i]), data['soft_judge_scores'][:, i].cuda()) for i in range(num_judges)])
    return loss


def get_dataloaders(args):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders


def main(dataloaders, i3d, joint_model,evaluator, base_logger, args):
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    criterion = nn.KLDivLoss()
    optimizer = torch.optim.Adam([*i3d.parameters()] +[*joint_model.parameters()]+ [*evaluator.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_best = 0
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best: {rho_best} at epoch {epoch_best}')

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []

            if split == 'train':
                i3d.train()
                joint_model.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                i3d.eval()
                joint_model.eval()
                evaluator.eval()
                torch.set_grad_enabled(False)

            for data in tqdm(dataloaders[split]):
                true_scores.extend(data['final_score'].numpy())
                videos = data['video'].cuda()
                videos.transpose_(1, 2)  # N, C, T, H, W

                batch_size, C, frames, H, W = videos.shape
                clip_feats = torch.empty(batch_size, 10, feature_dim).cuda()
                for i in range(9):
                    clip_feats[:, i] = i3d(videos[:, :, 10 * i:10 * i + 16, :, :]).squeeze(2)
                clip_feats[:, 9] = i3d(videos[:, :, -16:, :, :]).squeeze(2)

                # add text
                text = data["text"]
                joint_feats, mseloss = joint_model(clip_feats, text)

                probs = evaluator(joint_feats)
                preds = compute_score(probs, data)
                pred_scores.extend([i.item() for i in preds])

                if split == 'train':
                    klloss = compute_loss(criterion, probs, data)
                    loss = klloss + mseloss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rho, p = stats.spearmanr(pred_scores, true_scores)

            pred_scores = np.array(pred_scores)
            true_scores = np.array(true_scores)
            RL2 = 100 * np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
                  true_scores.shape[0]

            log_and_print(base_logger, f'{split} correlation: {rho} R-ℓ2: {RL2}')



        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            log_and_print(base_logger, '-----New best found!-----')
            log_and_print(base_logger, f'*******************pred_scores：{pred_scores}')
            log_and_print(base_logger, f'*******************true_scores：{true_scores}')

            path = './MTL-AQA/ckpts/' + str(rho) + '.pt'
            torch.save({'epoch': epoch,
                        'i3d': i3d.state_dict(),
                        "joint_model":joint_model.state_dict(),
                        'evaluator': evaluator.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rho_best': rho_best}, path)


if __name__ == '__main__':

    args = get_parser().parse_args()

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    init_seed(args)

    base_logger = get_logger(f'exp/MUSDL+LVFL.log', args.log_info)
    i3d, joint_model,evaluator = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, joint_model,evaluator, base_logger, args)
