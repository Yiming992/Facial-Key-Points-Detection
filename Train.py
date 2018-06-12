from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from Data import FacialKeypointsDataset
from Transform import Rescale,RandomCrop,Normalize,ToTensor
import argparse
import torch

data_transform=transforms.Compose([Rescale(250),RandomCrop(224),Normalize(),ToTensor()])

transformed_dataset=FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',root_dir='data/training/',transform=data_transform)
if __name__=='__main__':

    ap=argparse.ArgumentParser()
    ap.add_argument('-b','--batch_size',required=True)
    ap.add_argument('-e','--epochs',required=True)
    ap.add_argument('-s','--save',required=True,default=False)
    args=vars(ap.parse_args())

    train_loader=DataLoader(transformed_dataset,batch_size=args['batch_size'],shuffle=True,
                            num_workers=4)

    from model import Net
    import torch.nn as nn
    import torch.optim as optim
    net=Net()
 
    net.cuda()

    criterion=nn.MSELoss()

    optimizer=optim.Adam(net.parameters(),eps=0.1)

    for epoch in range(args['epochs']):

        running_loss=0.0

        for batch_i,data in enumerate(train_loader):

            images=data['image']
            key_pts=data['keypoints']

            key_pts=key_pts.view(key_pts.size(0),-1)

            key_pts=key_pts.type(torch.FloatTensor)
            images=images.type(torch.FloatTensor)

            key_pts=key_pts.cuda()
            images=images.cuda()

            output_pts=net(images)

            loss=criterion(output_pts,key_pts)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss+=loss.item()

            if batch_i %10 ==9:
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss=0.0
    print(Finished Training)

    if args['save']==True:

        model_dir='saved_models/'
        model_name='keypoints_model.pt'

        torch.save(net.state_dict(),model_dir+model_name)



    

