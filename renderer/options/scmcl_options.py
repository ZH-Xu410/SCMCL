from .train_options import TrainOptions as TrainOptions_


class TrainOptions(TrainOptions_):
    def initialize(self):
        TrainOptions_.initialize(self)
        self.parser.add_argument('--seq_len', type=int, default=10, help='Length of exp. coeffs. sequence')
        self.parser.add_argument('--hop_len', type=int, default=1, help='Hop Length (set to 1 by default for test)')
        self.parser.add_argument('--selected_emotions', type=str, nargs='+', help='Subset (or all) of the 8 basic emotions',
                                 default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'],
                                 choices=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'contempt'])
        self.parser.add_argument('--train_root', type=str, default='./MEAD_data', help='Directory containing (reconstructed) MEAD')
        self.parser.add_argument('--selected_actors', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003','M009','W029'])
        self.parser.add_argument('--selected_actors_ref', type=str, nargs='+', help='Subset of the MEAD actors', default=['M003','M009','W029'])
        self.parser.add_argument('--selected_actors_val', type=str, nargs='+', help='Subset of the MEAD actors', default=['M023'])
        self.parser.add_argument('--selected_actors_wild', type=str, nargs='+', help='Subset of YouTube actors', default=[])
        self.parser.add_argument('--dist_file', type=str, help='audio distances for training dataset', default=None)
        self.parser.add_argument('--dist_thresh', type=float, help='audio distances threshold', default=2.0)
        self.parser.add_argument('--margin', type=int, help='crop margin for image dist', default=70)

        self.parser.add_argument('--latent_dim', type=int, default=4, help='Latent vector dimension')
        self.parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of mapping network')
        self.parser.add_argument('--style_dim', type=int, default=16, help='Style code dimension')

        self.parser.add_argument('--manipulator_pretrain_weight', type=str, default='', help='manipulator pretrain weight')
        self.parser.add_argument('--encoder_ckpt', type=str, help='image encoder and scale layer checkpoint', default=None)
        self.parser.add_argument('--beta2', type=float, default=0.999, help='decay rate for 2nd moment of Adam')
        self.parser.add_argument('--g_lr', type=float, default=1e-5, help='initial learning rate for G network')
        self.parser.add_argument('--d_lr', type=float, default=1e-5, help='initial learning rate for D network')
        self.parser.add_argument('--e_lr', type=float, default=1e-5, help='initial learning rate for style_encoder')
        self.parser.add_argument('--m_lr', type=float, default=1e-5, help='initial learning rate for mapping_network')
        self.parser.add_argument('--rg_lr', type=float, default=1e-5, help='initial learning rate for renderer G network')
        self.parser.add_argument('--rd_lr', type=float, default=1e-5, help='initial learning rate for renderer D network')
        self.parser.add_argument('--lambda_cyc', type=float, default=1, help='Weight for cycle consistency loss')
        self.parser.add_argument('--lambda_sty', type=float, default=1, help='Weight for style reconstruction loss')
        self.parser.add_argument('--lambda_mouth', type=float, default=1, help='Weight for mouth loss')
        self.parser.add_argument('--lambda_paired', type=float, default=1, help='Weight for paired loss')
        self.parser.add_argument('--lambda_sim', type=float, default=1, help='Weight for similarity consistency loss')
        self.parser.add_argument('--manipulator_trainable', type=bool, default=False, help='wether manipulator trainable or not')
        self.parser.add_argument('--seed', type=int, default=777, help='random seed')
        self.parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    