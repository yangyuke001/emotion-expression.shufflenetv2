import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import opt
from ShuffleNetV2 import ShuffleNetV2
import torchvision


test_data = './test_img/test_fer/'
checkpoint = torch.load("./weights/ShuffleNetV2_0.98.pth")

def test(**kwargs):
    opt.parse(kwargs)
    import ipdb;
    ipdb.set_trace()
    # configure model
    ShuffleNetV2= ShuffleNetV2()
    ShuffleNetV2.load_state_dict(checkpoint)
    ShuffleNetV2.eval()

    if opt.use_gpu: ShuffleNetV2.cuda()
    

    # data
    test_data=torchvision.datasets.ImageFolder(test_data,
            transform=transforms.Compose(
                [transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]))


    
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False,
        num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in enumerate(test_dataloader):
        input = torch.autograd.Variable(data,volatile = True)
        if opt.use_gpu: input = input.cuda()
        score = ShuffleNetV2(input)
        probability = torch.nn.functional.softmax(score)[:,0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()
        
        batch_results = [(path_,probability_) for path_,probability_ in 
            zip(path,probability) ]

        results += batch_results
    write_csv(results,opt.result_file)

    return results