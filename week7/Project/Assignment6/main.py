import os

import torch
from PIL import Image
from timm import create_model
from collections import OrderedDict
from timm.models import resume_checkpoint

from src import *

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(args.analysis, list):
        methods = args.analysis
    elif args.analysis == 'all':
        methods = list(get_method_dict().keys())
    else:
        raise AttributeError(f'{args.analysis} is not supported')

    cnn_name = 'skresnext50_32x4d'
    model_cnn = create_model(
        'skresnext50_32x4d',
        num_classes=args.n_class).to(device)
    model_cnn.eval()

    # tf_name = 'dino'
    # model_tf = create_model('dino_vitsmall_patch8', pretrained=True).to(device)
    # model_tf.eval()

    checkpoint = torch.load('8.tar', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v
    model_cnn.load_state_dict(new_state_dict)

    print(f"To be performed: {methods}")

    data = transform(Image.open(args.img_path), args.test_size).to(device)

    # list_graph_node_names(model_cnn, verbose=False)

    common_kwargs = dict(data=data, label=args.img_label, img_size=args.test_size, data_dir=args.data_dir,
                         pca_data_dir=args.pca_data_dir, num_class=args.n_class, device=device, n_components=3,
                         ctype=['zoom_blur', 'contrast'], intensity=[1, 2])

    for m in methods:
        os.makedirs(args.output_dir + f'/{cnn_name}', exist_ok=True)
        # os.makedirs(args.output_dir + f'/{tf_name}', exist_ok=True)
        method = create_method(m)

        method(model=model_cnn, model_name=cnn_name, return_nodes=RETURN_NODES[cnn_name], attn_return_nodes='',
               save=args.output_dir + f'/{cnn_name}', **common_kwargs)
        clear()
        # method(model=model_tf, model_name=tf_name, return_nodes=RETURN_NODES[tf_name], patch_size=8,
        #        attn_return_nodes=ATTN_RETURN_NODES[tf_name], save=args.output_dir + f'/{tf_name}', **common_kwargs)
        # clear()
