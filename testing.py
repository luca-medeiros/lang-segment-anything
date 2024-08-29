from torchvision import datasets
from torchvision import transforms
import groundingdino.datasets.transforms as T
import os
from torch.utils.data import DataLoader
from lang_sam import LangSAM
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
import json
import lang_sam.utils as utils
from PIL import Image


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
def get_phrases_from_posmap(
    posmap, tokenized, tokenizer, left_idx=0, right_idx= 255
):
    assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
    if posmap.dim() == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")

def post_process(outputs,texts,idx,box_threshold=0.3,text_threshold=0.25,W=1200,H=800):
  prediction_logits = outputs["pred_logits"][idx][None].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
  prediction_boxes = outputs["pred_boxes"][idx][None].cpu()[0]  # prediction_boxes.shape = (nq, 4)
  mask = prediction_logits.max(dim=1)[0] > box_threshold
  logits = prediction_logits[mask]  # logits.shape = (n, 256)
  boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
  boxes = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
  tokenizer = model.groundingdino.tokenizer
  tokenized = tokenizer(texts[idx])
  phrases = [
              get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
              for logit
              in logits
          ]
  return boxes,phrases


transform= transforms.Compose([
        transforms.Resize((800,1200)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CustomDataset(Dataset):
    def __init__(self, root,annoations, transform=None, target_transform=None,no_obj=None):
        # Initializing Captcha Dataset class
        super(CustomDataset, self).__init__()
        self.root=root
        self.trans=transforms.Resize((800,1200))
        self.files = sorted(os.listdir(root))
        self.images = []
        self.captions = []
        self.transform=transform
        self.target_transform=target_transform
        df=pd.read_csv(annoations)
        df=df.set_index("image_ids_ref")
        files=os.listdir(root)
        for file in files:
          img_id=file.split(".")[0]
          img_id=int(img_id)
          cap=df.loc[img_id].iloc[0]["new_objects"]
          self.images.append(os.path.join(self.root, file))
          self.captions.append(self.preprocess_caption(cap))


    def __len__(self):
        # Returning the total number of images
        return len(self.captions)
    def preprocess_caption(self,caption):
      caption=caption.replace("[\'","").replace("\']","").replace("\', \'",". ")
      result = caption.lower().strip()
      if result.endswith("."):
          return result
      return result + "."
    def __getitem__(self, idx:int):
        # Retrieving image and target label based on index
        img, target = self.images[idx], self.captions[idx]
        raw_image=self.read_image(img)
        if self.transform is not None:
            img= self.transform(raw_image)
        else:
            img = raw_image

        # Applying target transformations if specified

        # Returning image and target label
        return img, target,np.asarray(self.trans(raw_image))

    # Method to read image from file path
    def read_image(self, path):
        img = Image.open(path)
        # Converting image to RGB format
        return img.convert('RGB')

    # Method to check if a file is an image file
    def is_image_file(self, filename):
        # Checking if filename ends with any of the specified image extensions
        return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png"])

dataset=CustomDataset('Coco_Subset',"new_objects.csv",
                      transform=transform
)

model = LangSAM()
torch.manual_seed(75)
loader=DataLoader(dataset,batch_size=1, shuffle=True, num_workers=2)
for img,cap,raw_img in loader:
  break
with torch.no_grad():
  img=img.to("cuda")
  outputs = model.groundingdino.to("cuda")(img,captions=cap)
  input_image_torch_list=[]
  for i in range(len(raw_img)):
    input_image = model.sam.transform.apply_image(np.transpose(raw_img[i],(2,0,1)))
    input_image_torch = torch.as_tensor(input_image)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_image_torch_list.append(input_image_torch)
  input_image_torch=torch.cat(input_image_torch_list,dim=0)
  model.sam.set_torch_image(input_image_torch.cuda(), raw_img.shape[1:3])
  transformed_boxes =model.sam.transform.apply_boxes_torch(outputs["pred_boxes"].cpu(), raw_img.shape[1:3])
  masks, _, _ = model.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(model.sam.device),
            multimask_output=False,
        )
#   if len(outputs["pred_boxes"]) > 0:
#           masks = self.predict_sam(image_pil, boxes)
#           masks = masks.squeeze(1)
# indx=0
# inv_normalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.225]
# )
# im_inv = inv_normalize(img[indx])
# boxes,phrases=post_process(outputs,cap,indx)
# new_image=utils.draw_image(np.transpose(im_inv.cpu().numpy(), (1, 2, 0)),masks=[],boxes=boxes,labels=phrases)
# print(cap[indx])
# print(phrases)
# plt.imshow(new_image)
