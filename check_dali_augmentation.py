#reference: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html

import nvidia.dali.fn as fn
import nvidia.dali as dali
from nvidia.dali.plugin.pytorch import feed_ndarray
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid
import torch
import os,glob
from PIL import Image


class ExternalInputGPUIterator(object):
    def __init__(self, images):
        self.images = 255*images.permute(0,2,3,1).contiguous() 

    def __iter__(self):
        self.i=0
        self.n=self.images.shape[0]
        return self

    def __next__(self):
        return [self.images[i,:,:,:].type(torch.uint8) for i in range(self.n)]



img_list = glob.glob('../../data/imagenet/n01440764/*.JPEG')

transform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])
tensors=[]
for i in range(25):
    img_path = img_list[i]
    img = Image.open(img_path).convert('RGB')
    tensors.append(transform(img).unsqueeze(dim=0))
styled_image = torch.cat(tensors, dim=0)
save_image(make_grid(styled_image, nrow=5), 'before.jpg')
print('max: {}, min: {}'.format(torch.max(styled_image), torch.min(styled_image)))
print(styled_image.shape)

eii = ExternalInputGPUIterator(styled_image.to('cuda'))
pipe = Pipeline(batch_size=25, num_threads=1, device_id=0)
with pipe:
    styled_image = fn.external_source(source=eii, device='gpu', batch=True, cuda_stream=0, dtype=types.UINT8)
    styled_image = fn.random_resized_crop(styled_image, size=96)
    styled_image = fn.flip(styled_image, horizontal=1, vertical=0) if torch.rand(1)<0.5 else styled_image
    b,c,s = torch.distributions.uniform.Uniform(1-0.8, 1+0.8).sample([3,])
    h = torch.distributions.uniform.Uniform(-0.2, 0.2).sample([1,])
    styled_image = fn.color_twist(styled_image, brightness=b, contrast=c, saturation=s, hue=h) if torch.rand(1)<0.8 else styled_image #only accept hwc
    styled_image = fn.color_space_conversion(styled_image, image_type=types.RGB, output_type=types.GRAY) if torch.rand(1)<0.2 else styled_image #only accept hwc, uint8
    styled_image = fn.gaussian_blur(styled_image, window_size=int(0.1*96))
    pipe.set_outputs(styled_image)

pipe.build()
styled_image=pipe.run()

# print(styled_image) # shpe:(1,) type:TensorListGPU
styled_image = styled_image[0].as_tensor() # type:TensorGPU
# print(styled_image)
styled_torch = torch.zeros(styled_image.shape(), dtype=torch.uint8).cuda()
feed_ndarray(styled_image, styled_torch)
styled_torch = styled_torch.permute(0,3,1,2).type(torch.float32)/225.
c=styled_torch.shape[1]
if c==1:
    styled_torch = styled_torch.repeat(1,3,1,1)

#gpu->cpu->gpu
# styled_torch = torch.cat([torch.Tensor(styled_image[0].as_cpu().at(i)).permute(2,0,1).unsqueeze(dim=0) for i in range(25)], dim=0)
# styled_torch = styled_torch.type(torch.float32)/225.

print(styled_torch.shape)
save_image(make_grid(styled_torch,nrow=5), 'after.jpg')