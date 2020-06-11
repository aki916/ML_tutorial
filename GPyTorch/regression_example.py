import math
import torch
import gpytorch
from matplotlib import pyplot as plt

def prepare_data(n):
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, n)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
    return train_x, train_y

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(model, optimizer, mll, train_x, train_y, training_iter):
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    return model

def test(xmin,xmax,model,likelihood):
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(xmin, xmax, 51)
        observed_pred = likelihood(model(test_x))
    return test_x, observed_pred

def plot(train_x, train_y, test_x, observed_pred):
    with torch.no_grad():
        # Get upper and lower confidence bounds(2 standard deviations)
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        plt.scatter(train_x.numpy(), train_y.numpy(), color='k', marker='*')
        # Plot predictive means as blue line
        plt.plot(test_x.numpy(), observed_pred.mean.numpy(), color='b')
        # Shade between the lower and upper confidence bounds
        plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        plt.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()


def main():
    n = 100
    train_x, train_y = prepare_data(n)
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    training_iter = 50


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # train
    model = train(model, optimizer, mll, train_x, train_y, training_iter)

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    xmin, xmax = -0.2, 1.2
    test_x, observed_pred = test(xmin,xmax,model,likelihood)
    plot(train_x, train_y, test_x, observed_pred)



if __name__ == '__main__':
    main()