import torch
import skimage.io as io
import numpy as np
import torchvision
from torchvision.models import vgg16
from torchvision.transforms import Compose,CenterCrop,Normalize,Scale
from torchvision.transforms import ToTensor,ToPILImage
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import SGD,Adam
from utils.datasets import  cyclegan_dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import cv2
from PIL import Image
from torch.nn import init



class Norm:
    def __call__(self,image):
        return torch.add(torch.mul(image,2),-1.0)

def weight_init_normal(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        init.uniform(m.weight.data,0.0,0.02)
    elif classname.find('Linear')!=-1:
        init.uniform(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm2d')!=-1:
        init.uniform(m.weight.data,1.0,0.02)
        init.constant(m.bias.data,0.0)

class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        
        self.conv=nn.Sequential(nn.Conv2d(3,16,3,2),
                                nn.LeakyReLU(inplace=True),
                                nn.BatchNorm2d(16),
                                nn.Conv2d(16,32,3,2),
                                nn.LeakyReLU(inplace=True),
                                nn.BatchNorm2d(32),          
                                nn.Conv2d(32,64,3,2),
                                nn.LeakyReLU(inplace=True),
                                nn.BatchNorm2d(64),
                                nn.Conv2d(64,128,3,4),
                                nn.LeakyReLU(inplace=True),
                                nn.BatchNorm2d(128),
                                nn.Conv2d(128,1,3,4),
                                )

    def forward(self,x):
        res=self.conv(x)
        return res.view(-1,1)
 
class G(nn.Module):
    def create_block(self,channel_in):
        return nn.Sequential(nn.Conv2d(channel_in,channel_in*2,4,2,padding=1),
                             nn.LeakyReLU(inplace=True),
                             nn.BatchNorm2d(channel_in*2),
                             nn.Conv2d(channel_in*2,channel_in*2,3,1,padding=1),
                             nn.LeakyReLU(inplace=True),
                             nn.BatchNorm2d(channel_in*2),
                             nn.Conv2d(channel_in*2,channel_in*2,3,1,padding=1),
                             nn.LeakyReLU(inplace=True),
                             nn.BatchNorm2d(channel_in*2),
                             )

    def merge_block(self,channel_in):
        return nn.Sequential(nn.Conv2d(channel_in,channel_in//2,3,1,padding=1),
                             nn.LeakyReLU(inplace=True),
                             nn.BatchNorm2d(channel_in//2),
                             nn.Conv2d(channel_in//2,channel_in//2,3,1,padding=1),
                             nn.LeakyReLU(inplace=True),
                             nn.BatchNorm2d(channel_in//2),
                             )

    def __init__(self):
        super(G,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(3,64,3,1,padding=1),
                                 nn.LeakyReLU(inplace=True),
                                 nn.BatchNorm2d(64)
                                 )
        self.block1=self.create_block(64)
        self.block2=self.create_block(128)
        self.block3=self.create_block(256) 
        self.block4=self.create_block(512) 
        self.deconv1=nn.ConvTranspose2d(1024,512,4,2,padding=1)
        self.deconv2=nn.ConvTranspose2d(512,256,4,2,padding=1)
        self.deconv3=nn.ConvTranspose2d(256,128,4,2,padding=1)
        self.deconv4=nn.ConvTranspose2d(128,64,4,2,padding=1)
        self.deblock1=self.merge_block(1024)
        self.deblock2=self.merge_block(512)
        self.deblock3=self.merge_block(256)
        self.deblock4=self.merge_block(128)
        self.conv2=nn.Sequential(nn.Conv2d(64,3,3,1,padding=1),
                                 nn.Tanh()
                                 )

    def forward(self,x):
        conv1=self.conv1(x)
        #print(conv1.data[0].size())
        block1=self.block1(conv1)
        block2=self.block2(block1)
        block3=self.block3(block2)    
        block4=self.block4(block3)    
        deblock1_=self.deblock1(torch.cat((block3,self.deconv1(block4)),dim=1))
        deblock2_=self.deblock2(torch.cat((block2,self.deconv2(deblock1_)),dim=1))
        deblock3_=self.deblock3(torch.cat((block1,self.deconv3(deblock2_)),dim=1))
        deblock4_=self.deblock4(torch.cat((conv1,self.deconv4(deblock3_)),dim=1))
        res=self.conv2(deblock4_)
        return res


D_1=D()
D_2=D()
D_1=D_1.cuda()
D_2=D_2.cuda()
D_1.apply(weight_init_normal)
D_2.apply(weight_init_normal)
D_1=torch.nn.DataParallel(D_1,device_ids=[0,1])
D_2=torch.nn.DataParallel(D_2,device_ids=[0,1])

G_1=G()
G_2=G()
G_1=G_1.cuda()
G_2=G_2.cuda()
G_1.apply(weight_init_normal)
G_2.apply(weight_init_normal)
G_1=torch.nn.DataParallel(G_1,device_ids=[0,1])
G_2=torch.nn.DataParallel(G_2,device_ids=[0,1])

data_dir1="CycleGAN/cityscapes/trainA/"
data_dir2="CycleGAN/cityscapes/trainB/"

input_transform=Compose([
                        Scale(128),
                        CenterCrop(128),
                        ToTensor(),
                        Norm()
])


def train(D_1,D_2,G_1,G_2,batch_size,epoches):
    D_1.train()
    D_2.train()
    G_1.train()
    G_2.train()

    loader=DataLoader(cyclegan_dataset(data_dir1,data_dir2,input_transform),
                      num_workers=4,batch_size=batch_size,shuffle=True)

    criterion_D=nn.MSELoss()
    criterion_G=nn.MSELoss()
    criterion_consistance=nn.L1Loss()

    optimizer_D_1=Adam(D_1.parameters(),2e-4,[0.5,0.999])
    optimizer_G_1=Adam(G_1.parameters(),2e-4,[0.5,0.999])
    optimizer_D_2=Adam(D_2.parameters(),2e-4,[0.5,0.999])
    optimizer_G_2=Adam(G_2.parameters(),2e-4,[0.5,0.999])
    
    true_label=torch.FloatTensor(batch_size).fill_(1.0)
    false_label=torch.FloatTensor(batch_size).fill_(0.0)
    true_label=true_label.cuda()
    false_label=false_label.cuda()
    true_label=Variable(true_label)
    false_label=Variable(false_label)
    fps=24
    fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
    vw=cv2.VideoWriter("demo/test.avi",fourcc,fps,(128*3,128))#same as show_image changing the width and height
    for epoch in range(1,epoches+1):
        for step,(images1,images2) in enumerate(loader):
            #print(step)
            #print(images1.size())
            #print(images2.size())
            images1=images1.cuda()
            images2=images2.cuda()

            inputs1=Variable(images1)
            inputs2=Variable(images2)
            
            d_res_1=D_1(inputs1)
            gen_image_1=G_1(inputs1)
            d_res_2=D_2(inputs2)
            gen_image_2=G_2(inputs2)
            d_fake_1=D_1(gen_image_2)
            d_fake_2=D_2(gen_image_1)

            loss_D_1_real=criterion_D(d_res_1,true_label)
            loss_D_1_fake=criterion_D(d_fake_1,false_label)
            loss_D_1=(loss_D_1_real+loss_D_1_fake)*0.5


            loss_D_2_real=criterion_D(d_res_2,true_label)
            loss_D_2_fake=criterion_D(d_fake_2,false_label)
            loss_D_2=(loss_D_2_real+loss_D_2_fake)*0.5
            
            loss_cycle_1=criterion_consistance(G_2(gen_image_1),inputs1)
            loss_cycle_2=criterion_consistance(G_1(gen_image_2),inputs2)
            
            loss_G_1=criterion_G(d_fake_1,true_label)+10*loss_cycle_2
            loss_G_2=criterion_G(d_fake_2,true_label)+10*loss_cycle_1
            
            optimizer_G_1.zero_grad()
            loss_G_1.backward(retain_graph=True)
            optimizer_G_1.step()
            
            optimizer_G_2.zero_grad()
            loss_G_2.backward(retain_graph=True)
            optimizer_G_2.step()

            optimizer_D_1.zero_grad()
            loss_D_1.backward(retain_graph=True)
            optimizer_D_1.step()

            optimizer_D_2.zero_grad()
            loss_D_2.backward(retain_graph=True)
            optimizer_D_2.step()
            
            if step%200==0:
                print("D_1 loss: ",loss_D_1.data[0],"G_1 loss: ",loss_G_1.data[0],"D_2 loss: ",loss_D_2.data[0],"G_2 loss: ",loss_G_2.data[0])
            if step%10==0:
                show_demo(G_1,G_2,epoch,vw)



def show_demo(model1,model2,i,videoWriter):
    raw_image=cv2.imread("demo/1.jpg")#must be same scale with the gan image
    raw_image=cv2.resize(raw_image,(128,128))
    image=Image.open("demo/1.jpg").convert('RGB')
    image=input_transform(image)
    image=image.cuda()
    gen_image=model1(Variable(image,volatile=True).unsqueeze(0))
    cycle_image=model2(gen_image)
    gen_image=gen_image.cpu()
    gen_image=gen_image.data.numpy()
    #print(label.shape)
    gen_image=np.squeeze(gen_image)
    gen_image=(gen_image+1)*127.5
    gen_image=gen_image.astype('uint8')
    gen_image=np.clip(gen_image,0,255)
    gen_image=np.einsum('kij->ijk',gen_image)
    cycle_image=cycle_image.cpu()
    cycle_image=cycle_image.data.numpy()
    cycle_image=np.squeeze(cycle_image)
    cycle_image=(cycle_image+1)*127.5
    cycle_image=cycle_image.astype('uint8')
    cycle_image=np.clip(cycle_image,0,255)
    cycle_image=np.einsum('kij->ijk',cycle_image)
    #print(raw_image.shape,gen_image.shape,cycle_image.shape)
    show_image=np.concatenate([raw_image,gen_image,cycle_image],axis=1)
    ####videoWriter
    #print(show_image.shape)
    videoWriter.write(show_image)

def save_weight(model,i):
    torch.save(model.state_dict(),f"outputs/{i}.pth")


train(D_1,D_2,G_1,G_2,1,100)

