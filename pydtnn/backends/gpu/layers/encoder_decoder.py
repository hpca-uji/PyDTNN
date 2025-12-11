from pydtnn.backends.gpu.libs import libcudnn as cudnn

from pydtnn.backends.gpu.layers.abstract.block_layer import AbstractBlockLayerGPU
from pydtnn.layers.encoder import Encoder
from pydtnn.layers.decoder import Decoder
from pydtnn.layers.encoder_decoder import EncoderDecoder


class EncoderDecoderGPU(AbstractBlockLayerGPU, EncoderDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = [Encoder(embedl=self.embedl, d_k=self.d_k, d_ff=self.d_ff, heads=self.heads, dropout_rate=self.dropout_rate) for _ in range(self.enc_layers)]
        self.decoder = [Decoder(embedl=self.embedl, d_k=self.d_k, d_ff=self.d_ff, heads=self.heads, dropout_rate=self.dropout_rate) for _ in range(self.dec_layers)]
        self.paths = [self.encoder + self.decoder]

        # The next attributes will be initialized later
        self.y = self.dx = None

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        if len(x) == 2:
            x_enc, x_dec = x
            mask_enc = mask_dec = None
            enc_shape = (prev_shape[0], ())
            dec_shape = (prev_shape[0], prev_shape[1], ())
        else:
            x_enc, mask_enc, x_dec, mask_dec = x
            enc_shape = (prev_shape[0], prev_shape[1])
            dec_shape = (prev_shape[2], prev_shape[0], prev_shape[3])
        self.embedl = x_enc.shape[-1]

        # Initialize all sublayers
        for layer in self.children:
            layer.init_backend_from_model(self.model)

        self.encoder[0].initialize(prev_shape=enc_shape, x=(x_enc, mask_enc))
        for layer in self.encoder[1:]:
            layer.initialize(prev_shape=enc_shape, x=(x_enc, mask_enc))
        for layer in self.decoder:
            layer.initialize(prev_shape=dec_shape, x=(x_dec, x_enc, mask_dec))

        for layer in self.children:
            self.fwd_time += layer.fwd_time
            self.bwd_time += layer.bwd_time
            self.nparams += layer.nparams

    def initialize_block_layer(self):
        pass

    def forward(self, x):
        if len(x) == 2:
            x, y = x
            x_mask = y_mask = None
        else:
            x, x_mask, y, y_mask = x
        for i in range(self.enc_layers):  # Encoding layers
            x = self.encoder[i].forward(x, x_mask)
        for i in range(self.dec_layers):  # Decoding layers
            y = self.decoder[i].forward(y, x, y_mask)
        self.y = y
        return self.y

    def backward(self, prev_dx):
        alpha, beta = 1.0, 1.0
        dx_tgt, dx_enc = self.decoder[self.dec_layers-1].backward(prev_dx)
        for i in range(self.dec_layers-1, 0, -1):  # Decoding layers
            dx_tgt, dx_2 = self.decoder[i-1].backward(dx_tgt)
            cudnn.cudnnAddTensor(self.model.cudnn_handle,
                                 alpha, dx_2.desc, dx_2.ptr,
                                 beta, dx_enc.desc, dx_enc.ptr)  # dx_enc += dx2
        for i in range(self.enc_layers, 0, -1):  # Enconding layers
            dx_enc = self.encoder[i-1].backward(dx_enc)
        if self.need_dx:
            return dx_tgt, dx_enc
