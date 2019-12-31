import torch


# linear
def hookfoo(mod,gradin,gradout):
    print('module =',mod)
    for i in range(len(gradin)):
        print('len(gradin) =',len(gradin))
        print('\ngradin[{}].shape ='.format(i),gradin[i].shape)
        print('gradin[{}] ='.format(i),gradin[i])

        # print('\n\n\n\n')
        # print('gradin =',gradin)
        # print('\n\n\n\n')

    # for i in range(len(gradout)):
    #     print('gradout[{}].shape ='.format(i),gradout[i].shape)
    #     print('gradout[{}] ='.format(i),gradout[i])

    return (gradin[0], torch.nn.ReLU()(gradin[1]), gradin[2])

x = 2*torch.ones(8,4)
x.requires_grad = True

n = torch.nn.Linear(4,5)
for p in n.parameters():
    p.data = -1*torch.ones_like(p.data)

h = n.register_backward_hook(hookfoo)

y = n(x)
y.sum().backward()
print('\ny =',y)
print('x.grad =',x.grad)
print('\n\n\n\n')


# multi-input multi-output
class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo,self).__init__()

        self.fc1 = torch.nn.Linear(4,2)
        self.fc2 = torch.nn.Linear(5,3)

    def forward(self,x1,x2):
        return self.fc1(x1),self.fc2(x2)

f = Foo()
x1 = torch.ones(1,4)*3
x2 = torch.ones(1,5)*4
x1.requires_grad = True
x2.requires_grad = True

h = f.register_backward_hook(hookfoo)

y,z = f(x1,x2)
(y.sum() + z.sum()).backward()
print('x1.grad =',x1.grad)
print('x2.grad =',x2.grad)

print('\n\n\n\n')


# conv layers
def hookfoo(a,b,c):
    print('b[0].shape =',b[0].shape)
    print('b[0] =',b[0])

x = torch.rand(2,3,4,4,requires_grad=True)
print("x =",x)
n = torch.nn.Conv2d(3,5,3)
h = n.register_backward_hook(hookfoo)
n(x).sum().backward()
print('x.grad =',x.grad)
