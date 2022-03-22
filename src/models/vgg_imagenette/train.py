import argparse
import os
import mlflow
import mlflow.pytorch
import pickle
import tempfile
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from src.data.datasets import Imagenette
from src.models.vgg_imagenette import vgg
from tensorboardX import SummaryWriter

os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "AxhhIVxMi8sfwykiOgbt5gHCUdexWuLmbMCz6nRH"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.in.chireiden.moe"
mlflow.set_tracking_uri("https://mlflow.in.chireiden.moe")
# Command-line arguments
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument("--data-path",
                    type=str,
                    default="../../../data/raw/imagenette2-320/",
                    help="path to imagenette data root, i.e. /some/path/imagenette2-320/"
                    )
parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    metavar="N",
    help="input batch size for training (default: 16)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=8,
    metavar="N",
    help="input batch size for testing (default: 8)",
)
parser.add_argument(
    "--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)"
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
)
parser.add_argument(
    "--enable-cuda",
    type=str,
    choices=["True", "False"],
    default="True",
    help="enables or disables CUDA training",
)
parser.add_argument("--seed", type=int, default=1,
                    metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--image-dimensions",
    type=int,
    default=160,
    help="width and height (square image) of the input images (resized if too big/small)",
)
parser.add_argument("--n-classes", type=int, default=10,
                    help="number of classes (default: 10)")

args = parser.parse_args()

enable_cuda_flag = True if args.enable_cuda == "True" else False

args.cuda = enable_cuda_flag and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}

dims = (args.image_dimensions,) * 2

common_transforms = transforms.Compose(
    [
        transforms.Resize(dims),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

train_data = Imagenette(args.data_path + 'noisy_imagenette.csv',
                        args.data_path, train=True, transform=common_transforms)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

test_data = Imagenette(args.data_path + 'noisy_imagenette.csv',
                       args.data_path, train=False, transform=common_transforms)

test_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
model = vgg.vgg(conv_arch, in_channels=3,
                in_dims=dims, n_classes=args.n_classes)

if args.cuda:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'
model.to(device)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


model.apply(init_weights)


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()

writer = None


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
            step = epoch * len(train_loader) + batch_idx
            log_scalar("train_loss", loss.data.item(), step)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )
    step = (epoch + 1) * len(train_loader)
    log_scalar("test_loss", test_loss, step)
    log_scalar("test_accuracy", test_accuracy, step)


def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)


with mlflow.start_run():
    # Log our parameters into mlflow
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    # Create a SummaryWriter to write TensorBoard events locally
    output_dir = dirpath = tempfile.mkdtemp()
    writer = SummaryWriter(output_dir)
    print("Writing TensorBoard events locally to %s\n" % output_dir)

    # Perform the training
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

    # Upload the TensorBoard event logs as a run artifact
    print("Uploading TensorBoard events as a run artifact...")
    mlflow.log_artifacts(output_dir, artifact_path="events")
    print(
        "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
        % os.path.join(mlflow.get_artifact_uri(), "events")
    )

    # Log the model as an artifact of the MLflow run.
    print("\nLogging the trained model as a run artifact...")
    mlflow.pytorch.log_model(
        model, artifact_path="pytorch-model", pickle_module=pickle)
    print(
        "\nThe model is logged at:\n%s" % os.path.join(
            mlflow.get_artifact_uri(), "pytorch-model")
    )

    # Since the model was logged as an artifact, it can be loaded to make predictions
    loaded_model = mlflow.pytorch.load_model(
        mlflow.get_artifact_uri("pytorch-model"))

    # Extract a few examples from the test dataset to evaulate on
    eval_data, eval_labels = next(iter(test_loader))

    # Make a few predictions
    predictions = loaded_model(eval_data.to(device)).data.max(1)[1]
    template = 'Sample {} : Ground truth is "{}", model prediction is "{}"'
    print("\nSample predictions")
    for index in range(5):
        print(template.format(index, eval_labels[index], predictions[index]))
