from data import DoomImage
import numpy as np
from torch.utils.data import DataLoader
import tensorflow as tf
from tqdm import tqdm
from imageio import imwrite as imsave
from data import Places, PlacesRoom, PlacesOutdoor
from habitat_baselines.rl.models.resnet import ResNetEncoder
import habitat_baselines.rl.models.resnet as resnet
import torch
from gym.spaces import Box
from gym import spaces
import torch
torch.manual_seed(1)


if __name__ == "__main__":
    resnet_baseplanes = 32
    backbone = "resnet50"

    rgb_box = Box(0, 255, (256, 256, 3))

    model = ResNetEncoder(
        spaces.Dict({"rgb": rgb_box}),
        baseplanes=resnet_baseplanes*2,
        ngroups=resnet_baseplanes // 2,
        make_backbone=getattr(resnet, backbone),
        normalize_visual_inputs=True,
        obs_transform=None,
        backbone_only=True,
        dense=True
    )

    ckpt = torch.load("/private/home/yilundu/sandbox/habitat/habitat-lab/checkpoints/curiosity_pointnav_pretrain_resnet50_301_resume/curiosity_pointnav_pretrain/curiosity_pointnav_pretrain.16.pth")
    state_dict = ckpt['state_dict']
    weights_new = {}

    for k, v in state_dict.items():
        split_layer_name = k.split(".")[2:]

        if len(split_layer_name) == 0:
            continue

        if "model_encoder" == split_layer_name[0]:
            layer_name = ".".join(split_layer_name[1:])
            weights_new[layer_name] = v

    # model.load_state_dict(weights_new, strict=False)

    model = model.cuda()
    model = model.eval()

    train_data = PlacesRoom(train=True)
    test_data = PlacesRoom(train=False)

    train_dataloader = DataLoader(train_data, batch_size=512, num_workers=24, shuffle=True, drop_last=True)

    with torch.no_grad():
        for i, (im, _) in  enumerate(tqdm(train_dataloader)):
            im = im * 255
            im = im.float()
            im = im.cuda()

            embed_val = model({'rgb': im})
            embed_val = embed_val.mean(dim=2).mean(dim=2).detach().cpu().numpy()
            im = im.cpu()

            if i == 0:
                start_im = im[:64].numpy().copy()
                start_embed = embed_val[:64].copy()

                goal_im = np.zeros((5, *start_im.shape))
                embed_dist = np.ones((5, goal_im.shape[1])) * 1000
            else:
                dist_bulk = np.linalg.norm(embed_val[None, :] - start_embed[:, None], axis=2)

                for i in range(64):
                    dist_i = dist_bulk[i]

                    for j, dist_im in enumerate(dist_i):
                        if dist_im < embed_dist[:, i].max():
                            replace_idx = embed_dist[:, i].argsort()[-1]

                            embed_dist[replace_idx, i] = dist_im
                            goal_im[replace_idx, i] = im[j]

            # if i > 2:
            #     break


    # Start im is shape 64 x 256 x 256 x 3 while goal image is size 5 x 64 x 256 x 256 x 3
    goal_im = goal_im.transpose((1, 2, 0, 3, 4)).reshape((64, 256, 5*256, 3))
    panel_im = np.concatenate([start_im, goal_im], axis=2)
    panel_im = panel_im.reshape((64*256, 256*6, 3))
    imsave("habitat_near_random.png", panel_im)
