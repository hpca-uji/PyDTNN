import sys
import models.alexnet as alexnet
import models.vgg16 as vgg16
import models.inceptionv3 as inceptionv3
import models.resnet50 as resnet50
import models.simplecnn as simplecnn

def create_model(nn, comm):
    if nn == "vgg16":
        model = vgg16.create_vgg16(comm)
    elif nn == "alexnet":
        model = alexnet.create_alexnet(comm)
    elif nn == "inceptionv3":
       model = inceptionv3.create_inceptionv3(comm)
    elif nn == "resnet50":
        model = resnet50.create_resnet50(comm)
    elif nn == "simplecnn":
        model = simplecnn.create_simplecnn(comm)
    else:
        print("Model %s not found!" % nn)
        sys.exit(-1)
    return model

