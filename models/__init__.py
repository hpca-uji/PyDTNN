import sys
import models.alexnet as alexnet
import models.vgg16 as vgg16
import models.vgg11 as vgg11
import models.inceptionv3 as inceptionv3
import models.resnet50 as resnet50
import models.simplecnn as simplecnn
import models.simplemlp as simplemlp

def create_model(args, comm):
    if args.model == "vgg16":
        model = vgg16.create_vgg16(comm, args.tracing)
    elif args.model == "vgg11":
        model = vgg11.create_vgg11(comm, args.tracing)
    elif args.model == "alexnet":
        model = alexnet.create_alexnet(comm, args.tracing)
    elif args.model == "inceptionv3":
       model = inceptionv3.create_inceptionv3(comm, args.tracing)
    elif args.model == "resnet50":
        model = resnet50.create_resnet50(comm, args.tracing)
    elif args.model == "simplecnn":
        model = simplecnn.create_simplecnn(comm, args.tracing)
    elif args.model == "simplemlp":
        model = simplemlp.create_simplemlp(comm, args.tracing)
    else:
        print("Model %s not found!" % args.model)
        sys.exit(-1)
    return model

