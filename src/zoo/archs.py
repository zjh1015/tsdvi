import torch
from torch import nn
from torch.utils.data import dataset



class GaussianParametrizer(nn.Module):
    """ Linear mapper from Image features to mean and log-variance parameters of latent-space gaussian distribution. """

    def __init__(self,
                 latent_dim: int,
                 feature_dim: int,
                 args,
                 act_fn: object = nn.ReLU):
        """
        Inputs:
            - latent_dim : Dimensionality of latent representation z
            - feature_dim: dimensionality of the input feature
            - act_fn : Activation function used throughout the network (if at all)
        """
        super(GaussianParametrizer, self).__init__()

        self.args = args
        self.h1 = nn.Linear(feature_dim, latent_dim)
        self.h2 = nn.Linear(feature_dim, latent_dim)

    def forward(self, x):
        mu = self.h1(x)
        log_var = self.h2(x)
        return mu, log_var


class CEncoder(nn.Module):
    """ Convolutional Encoder to transform an input image into its flattened feature embedding. """

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 args,
                 act_fn: object = nn.LeakyReLU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers use 2x of it.
            - act_fn : Activation function used throughout the encoder network
        """
        super(CEncoder, self).__init__()
        c_hid = base_channel_size
        self.args = args

        act_fn = nn.ReLU() if (self.args.dataset == 'tiered' and self.args.k_shots == 1) else nn.LeakyReLU(0.2)
            
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            act_fn,
            nn.MaxPool2d(2),  # 28 x 28, # 42 x 42

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            act_fn,
            nn.MaxPool2d(2),  # 9x9 # 21 x 21

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            act_fn,
            nn.MaxPool2d(2),  # 3x3 # 10 x 10

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            act_fn,
            nn.MaxPool2d(2),  # 1x1 # 5 x 5
            nn.Flatten()
        )

    def forward(self, x):
        x = self.net(x)

        return x


class TADCEncoder(nn.Module):
    """ Convolutional Encoder to transform an input image into its task/episode aware feature embedding. 
        TAsk Dependent Convolutional Encoder--TADCEncoder """

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 args,
                 act_fn: object = nn.LeakyReLU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers use 2x of it.
            - args: dict of arguments
            - act_fn : Activation function used throughout the encoder network
        """

        super(TADCEncoder, self).__init__()
        c_hid = base_channel_size
        self.args = args

        act_fn = nn.ReLU() if (self.args.dataset == 'tiered' and self.args.k_shots == 1) else nn.LeakyReLU(0.2)

        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            act_fn,
            nn.MaxPool2d(2),  # 28 x 28, # 42 x 42

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            act_fn,
            nn.MaxPool2d(2),  # 9x9 # 21 x 21

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            act_fn,
            nn.MaxPool2d(2),  # 3x3 # 10 x 10

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hid),
            act_fn,
            nn.MaxPool2d(2)  # 1x1 # 5 x 5
        )

        self.n = args.n_ways * (args.k_shots + args.q_shots)
        self.fc = nn.Sequential(
            nn.Linear(64, base_channel_size * 50),
            act_fn,
            nn.Linear(base_channel_size * 50, base_channel_size * 25),

        )

        ## AttFEX Module ##
        # 1x1 Convs representing M(.), N(.)
        self.fe = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.args.wm_channels, kernel_size=(  # 64 --> args.wm
                self.n, 1), stride=(1, 1), padding='valid', bias=False),

            act_fn,
            nn.Conv2d(in_channels=self.args.wm_channels, out_channels=self.args.wn_channels, kernel_size=(  # 64 --> args.wm, 32 --> args.wn
                1, 1), stride=(1, 1), padding='valid', bias=False),

            act_fn)

        # Query, Key and Value extractors as 1x1 Convs
        self.f_q = nn.Conv2d(in_channels=self.args.wn_channels, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)
        self.f_k = nn.Conv2d(in_channels=self.args.wn_channels, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)
        self.f_v = nn.Conv2d(in_channels=self.args.wn_channels, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)

    def forward(self, x,z_a, update: str):
        # Removing semantically agnostic information from Images
        x = self.net(x)
        z_a = self.fc(z_a).reshape(-1, 32, 5, 5)
        x = x - z_a

        # using AttFEX
        G = x.permute(2, 3, 0, 1)
        G = G.reshape(G.shape[0] * G.shape[1],
                      G.shape[2], G.shape[3]).unsqueeze(dim=1)
        G = self.fe(G)

        if (self.args.dataset == 'tiered' and self.args.k_shots == 5) or (self.args.dataset == 'miniimagenet' and self.args.k_shots == 5):
            xq = self.f_q(G)
            xq = nn.LeakyReLU(0.2)(xq)
            xk = self.f_k(G)
            xk = nn.LeakyReLU(0.2)(xk)
            xv = self.f_v(G)
            xv = nn.LeakyReLU(0.2)(xv)
        else:
            xq = self.f_q(G)
            xk = self.f_k(G)
            xv = self.f_v(G)

        xq = xq.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x.shape[2], x.shape[3])
        xk = xk.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x.shape[2], x.shape[3])
        xv = xv.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x.shape[2], x.shape[3])

        # Attention Block
        xq = xq.reshape(xq.shape[0], xq.shape[1]*xq.shape[2])
        xk = xk.reshape(xk.shape[0], xk.shape[1]*xk.shape[2])
        xv = xv.reshape(xv.shape[0], xv.shape[1]*xv.shape[2])

        G = torch.mm(xq, xk.transpose(0, 1)/xk.shape[1]**0.5)
        softmax = nn.Softmax(dim=-1)
        G = softmax(G)
        G = torch.mm(G, xv)

        # Transductive Mask transformed input
        G = G.reshape(-1, x.shape[2], x.shape[3])
        if update == 'inner':
            x = x[:self.args.n_ways*self.args.k_shots] * G
        elif update == 'outer':
            x = x[self.args.n_ways*self.args.k_shots:] * G

        x = nn.Flatten()(x)

        return x


class CDecoder(nn.Module):
    """ Convolutional Decoder for reconstructing an image using a latent variable z as input. """

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 args,
                 act_fn: object = nn.LeakyReLU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct.
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers use a 2x of it.
            - latent_dim : Dimensionality of latent representation z + Dimensionality of one-hot encoded label 
            - act_fn : Activation function used throughout the decoder network
        """
        super(CDecoder, self).__init__()
        c_hid = base_channel_size
        self.dataset = dataset
        act_fn = nn.ReLU() if (args.dataset == 'tiered' and args.k_shots == 1) else nn.LeakyReLU(0.2)
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 25*c_hid),
            act_fn
        )
        a1 = 10
        a2 = 21
        a3 = 42
        a4 = 84

        if (args.dataset == 'miniimagenet' and args.k_shots == 5) or (args.dataset == 'tiered' and args.k_shots == 1):
            self.net = nn.Sequential(

                nn.UpsamplingNearest2d(size=(a1, a1)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                        kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn,

                nn.UpsamplingNearest2d(size=(a2, a2)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                        kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn,

                nn.UpsamplingNearest2d(size=(a3, a3)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                        kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_hid),
                act_fn,

                nn.UpsamplingNearest2d(size=(a4, a4)),
                nn.Conv2d(in_channels=c_hid, out_channels=num_input_channels,
                        kernel_size=3, padding='same'),
                nn.BatchNorm2d(num_input_channels),
                nn.Sigmoid()
            )

        else:
            self.net = nn.Sequential(

                nn.UpsamplingNearest2d(size=(a1, a1)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                        kernel_size=3, padding='same'),
                #nn.BatchNorm2d(c_hid),
                act_fn,

                nn.UpsamplingNearest2d(size=(a2, a2)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                        kernel_size=3, padding='same'),
                #nn.BatchNorm2d(c_hid),
                act_fn,

                nn.UpsamplingNearest2d(size=(a3, a3)),
                nn.Conv2d(in_channels=c_hid, out_channels=c_hid,
                        kernel_size=3, padding='same'),
                #nn.BatchNorm2d(c_hid),
                act_fn,

                nn.UpsamplingNearest2d(size=(a4, a4)),
                nn.Conv2d(in_channels=c_hid, out_channels=num_input_channels,
                        kernel_size=3, padding='same'),
                #nn.BatchNorm2d(num_input_channels),
                nn.Sigmoid()
            )


    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 5, 5)
        x = self.net(x)
        return x





class Classifier_VAE(nn.Module):
    """ Module for a Convolutional-VAE: Convolutional Encoder + Linear Classifier that 
    transforms an input image into latent-space gaussian distribution, and uses z_l drawn 
    from this distribution to produce logits for classification. """

    def __init__(self, in_channels, base_channels, latent_dim_l, latent_dim_a, n_ways, args, act_fn: object = nn.LeakyReLU):
        super(Classifier_VAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim_l = latent_dim_l
        self.latent_dim_a = latent_dim_a
        self.classes = n_ways

        fcoeff = 25
        fsize = fcoeff*self.base_channels


        self.encoder = TADCEncoder(num_input_channels=self.in_channels,
                                       base_channel_size=self.base_channels, args=args)


        self.gaussian_parametrizer = GaussianParametrizer(
            latent_dim=self.latent_dim_l, feature_dim=fsize , args=args)

        act_fn = nn.ReLU() if (args.dataset == 'tiered' and args.k_shots == 1) else nn.LeakyReLU(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim_l, self.latent_dim_l//2), act_fn,
            nn.Linear(self.latent_dim_l//2, self.classes))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, z_a, update):

        x = self.encoder(x,z_a, update)

        mu_l, log_var_l = self.gaussian_parametrizer(x)
        z_l = self.reparameterize(mu_l, log_var_l)


        logits = self.classifier(z_l)
        return logits, mu_l, log_var_l, z_l


class CCVAE(nn.Module):
    """ Module for a Conditional-Convolutional VAE: Classifier-VAE + Convolutional Encoder-Decoder. 
    The Conv. Encoder-Decoder is conditioned on the z_l drawn from the class-latent gaussian distribution 
    for reconstructing the input image.
    The Classifier VAE is conditioned on the z_a drawn from the semantic-latent gaussian distribution for inference of z_l. """

    def __init__(self, in_channels, base_channels, n_ways,  args, latent_dim_l, latent_dim_a):
        super(CCVAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim_l = latent_dim_l
        self.latent_dim_a = latent_dim_a
        self.classes = n_ways
        self.args = args

        fcoeff = 25
        fsize = fcoeff*self.base_channels

        self.encoder = CEncoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, args=args)

        self.decoder = CDecoder(num_input_channels=self.in_channels,
                                base_channel_size=self.base_channels, latent_dim=(self.latent_dim_a + self.latent_dim_l), args=args)

        self.classifier_vae = Classifier_VAE(
            in_channels=self.in_channels, base_channels=self.base_channels, latent_dim_l=self.latent_dim_l, latent_dim_a=self.latent_dim_a, n_ways=self.classes, args=self.args)

        self.gaussian_parametrizer = GaussianParametrizer(
            latent_dim=self.latent_dim_a, feature_dim=fsize, args=args)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, update):

        xs = self.encoder(x)

        mu_a, log_var_a = self.gaussian_parametrizer(xs)
        z_a = self.reparameterize(mu_a, log_var_a)
        del xs

        logits, mu_l, log_var_l, z_l = self.classifier_vae(x, z_a, update)

        if update == 'inner':
            z_a = z_a[:self.args.n_ways*self.args.k_shots]
            mu_a = mu_a[:self.args.n_ways*self.args.k_shots]
            log_var_a=log_var_a[:self.args.n_ways*self.args.k_shots]
        elif update == 'outer':
            z_a = z_a[self.args.n_ways*self.args.k_shots:]
            mu_a = mu_a[self.args.n_ways * self.args.k_shots:]
            log_var_a = log_var_a[self.args.n_ways * self.args.k_shots:]

        x = self.decoder(torch.cat([z_a, z_l], dim=1))
        return x, logits, mu_l, log_var_l, mu_a, log_var_a