import torch

def hookfoo(mod,gradin,gradout):
    print('module =',mod)
    for i in range(len(gradin)):
        print('\ngradin[{}].shape ='.format(i),gradin[i].shape)
        print('gradin[{}] ='.format(i),gradin[i])

    # for i in range(len(gradout)):
    #     print('gradout[{}].shape ='.format(i),gradout[i].shape)
    #     print('gradout[{}] ='.format(i),gradout[i])

x = torch.ones(8,4,requires_grad=True)*2

n = torch.nn.Linear(4,5)
for p in n.parameters():
    p.data = torch.ones_like(p.data)

h = n.register_backward_hook(hookfoo)

y = n(x)
print('y =',y)
print('\n\n\n\n')
y.sum().backward()

print('x.grad =',x.grad)

print('\n\n\n\n')

class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo,self).__init__()

        self.fc1 = torch.nn.Linear(4,2)
        self.fc2 = torch.nn.Linear(5,3)

    def forward(self,x1,x2):
        return self.fc1(x1),self.fc2(x2)

f = Foo()
x1 = torch.ones(1,4,requires_grad=True)*3
x2 = torch.ones(1,5,requires_grad=True)*4

h = f.register_backward_hook(hookfoo)

y,z = f(x1,x2)
(y.sum() + z.sum()).backward()
