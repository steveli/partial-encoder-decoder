import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from fid import BaseSampler, BaseImputationSampler
from masked_celeba import BlockMaskedCelebA, IndepMaskedCelebA
from celeba_fid import CelebAFID
from celeba_decoder import ConvDecoder
from celeba_encoder import ConvEncoder
from celeba_pbigan import PBiGAN


parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--only', action='store_true')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Sampler(BaseSampler):
    def __init__(self, model, latent_size, images=60000, batch_size=256):
        super().__init__(images)
        self.model = model
        self.rand_z = torch.empty(batch_size, latent_size, device=device)

    def sample(self):
        self.rand_z.normal_()
        return self.model.decoder(self.rand_z)[1]


class ImputationSampler(BaseImputationSampler):
    def __init__(self, data_loader, model, batch_size=256):
        super().__init__(data_loader)
        self.model = model

    def impute(self, data, mask):
        z = self.model.encoder(data * mask)
        x_recon = self.model.decoder(z)[1]
        imputed_data = data * mask + x_recon * (1 - mask)
        return imputed_data


class Data:
    def __init__(self, args, batch_size):
        self.args = args
        self.batch_size = batch_size
        self.data_loader = None

    def gen_data(self):
        args = self.args
        if args.mask == 'indep':
            data = IndepMaskedCelebA(obs_prob=args.obs_prob)
        elif args.mask == 'block':
            data = BlockMaskedCelebA(block_len=args.block_len)

        self.data_size = len(data)
        self.data_loader = DataLoader(data, batch_size=self.batch_size)

    def get_data(self):
        if self.data_loader is None:
            self.gen_data()
        return self.data_loader, self.data_size


def pretrained_misgan_fid(model_file, data_loader, data_size):
    model = torch.load(model_file, map_location='cpu')

    model_args = model['args']
    decoder = ConvDecoder(model_args.latent)
    encoder = ConvEncoder(model_args.latent, model_args.flow, logprob=False)
    pbigan = PBiGAN(encoder, decoder, model_args.aeloss).to(device)
    pbigan.load_state_dict(model['pbigan'])

    batch_size = args.batch_size

    pbigan.eval()
    with torch.no_grad():
        compute_fid = CelebAFID(batch_size=batch_size)
        sampler = Sampler(pbigan, model_args.latent, data_size, batch_size)
        gen_fid = compute_fid.fid(sampler, data_size)
        print('fid: {:.2f}'.format(gen_fid))

        imputation_sampler = ImputationSampler(data_loader, pbigan, batch_size)
        imp_fid = compute_fid.fid(imputation_sampler, data_size)
        print('impute fid: {:.2f}'.format(imp_fid))

    return gen_fid, imp_fid


def gen_fid_file(model_file, fid_file, imp_fid_file, data):
    if imp_fid_file.exists():
        print('skip')
        return

    fid, imp_fid = pretrained_misgan_fid(model_file, *data.get_data())

    with fid_file.open('w') as f:
        print(fid, file=f)

    if imp_fid is not None:
        with imp_fid_file.open('w') as f:
            print(imp_fid, file=f)


def main():
    root_dir = Path(args.root_dir)
    model_file = root_dir / 'model.pth'
    print(model_file)
    fid_file = root_dir / 'fid.txt'
    imp_fid_file = root_dir / 'impute-fid.txt'

    model = torch.load(model_file, map_location='cpu')
    data = Data(model['args'], args.batch_size)

    gen_fid_file(model_file, fid_file, imp_fid_file, data)

    if args.only:
        return

    model_dir = root_dir / 'model'
    for model_file in sorted(model_dir.glob('*.pth')):
        print(model_file)
        fid_file = model_dir / f'{model_file.stem}-fid.txt'
        imp_fid_file = model_dir / f'{model_file.stem}-impute-fid.txt'
        gen_fid_file(model_file, fid_file, imp_fid_file, data)


if __name__ == '__main__':
    main()
