import scipy.constants as constants
import numpy
import math
#2D TE FDTD加PML仿真代码
c=constants.c #光速
mu_0=constants.mu_0 #真空中的磁导系数
epsilon_0=constants.epsilon_0     #真空中的介电系数
maxTime=300 #最大时间步长
sizeX, sizeY = (100, 100)
slen, shment=(40,50); #激励源脉冲宽度，脉冲峰值时刻
ez = numpy.zeros((sizeX,sizeY,maxTime))
ezx = numpy.zeros((sizeX,sizeY,maxTime))
ezy = numpy.zeros((sizeX,sizeY,maxTime))
hx = numpy.zeros((sizeX,sizeY,maxTime))
hy = numpy.zeros((sizeX,sizeY,maxTime)) #初始化场量
dt=1e-1 #时间步长
dx = 2 ** 0.5 * c * dt #参考《电磁波时域有限方法（第二版）》P.37
dy = 2 ** 0.5 * c * dt #参考《电磁波时域有限方法（第二版）》P.37
#初始化介电系数和磁导率矩阵
#epsilon=epsilon_0*numpy.ones((sizeX,sizeY,maxTime));
#mu=mu0*numpy.ones((sizeX,sizeY,maxTime));
epsilon=epsilon_0
mu=mu_0
#初始化电导率矩阵
sigmax=numpy.zeros((sizeX,sizeY))
sigmay=numpy.zeros((sizeX,sizeY))
#初始化磁导率矩阵
sigma_starx=numpy.zeros((sizeX,sizeY))
sigma_stary=numpy.zeros((sizeX,sizeY))
#PML的厚度
boundWidth=25;
#电导率多项式衰减模型中多项式的级数
gradingOrder=6;
#想要达到的反射系数
reflCoeff=1e-6;
#电导率的多项式模型
sigmaMax=(-math.log(reflCoeff)*(gradingOrder+1)*epsilon_0*c)/(2*boundWidth*dx);
boundfactor1=((epsilon/epsilon_0)*sigmaMax)/((boundWidth**gradingOrder)*(gradingOrder+1));
boundfactor2=((epsilon/epsilon_0)*sigmaMax)/((boundWidth**gradingOrder)*(gradingOrder+1));
boundfactor3=((epsilon/epsilon_0)*sigmaMax)/((boundWidth**gradingOrder)*(gradingOrder+1));
boundfactor4=((epsilon/epsilon_0)*sigmaMax)/((boundWidth**gradingOrder)*(gradingOrder+1));
for i in range(0,sizeX):
    for x in range(0,boundWidth):
        sigmax[i,boundWidth-x]=boundfactor1*((x+0.5)**(gradingOrder+1)-(x-0.5)**(gradingOrder+1))
        sigmax[i,sizeY-boundWidth+x]=boundfactor2*((x+0.5)**(gradingOrder+1)-(x-0.5)**(gradingOrder+1))
for i in range(0,sizeY):
    for x in range(0,boundWidth):
        sigmay[boundWidth-x,i]=boundfactor3*((x+0.5)**(gradingOrder+1)-(x-0.5)**(gradingOrder+1))
        sigmay[sizeX-boundWidth+x,i]=boundfactor4*((x+0.5)**(gradingOrder+1)-(x-0.5)**(gradingOrder+1))
sigma_starx= (sigmax*mu)/epsilon
sigma_stary=(sigmay*mu)/epsilon
for currentTime in range(1,maxTime):  # 在时间步长里迭代
   for x in range(1,sizeX):
        for y in range(1,sizeY):
            hy[x,y,currentTime]=((mu-0.5*dt*sigma_stary[x,y])/((mu+0.5*dt*sigma_stary[x,y])))*hy[x,y,currentTime-1]+((dt/dx)/(mu+0.5*dt*sigma_stary[x,y]))*(ezx[x,y-1,currentTime-1]-ezx[x-1,y-1,currentTime-1]+ezy[x,y-1,currentTime-1]-ezy[x-1,y-1,currentTime-1])
            hx[x,y,currentTime]=((mu-0.5*dt*sigma_starx[x,y])/((mu+0.5*dt*sigma_starx[x,y])))*hx[x,y,currentTime-1]-((dt/dy)/(mu+0.5*dt*sigma_starx[x,y]))*(ezx[x-1,y,currentTime-1]-ezx[x-1,y-1,currentTime-1]+ezy[x-1,y,currentTime-1]-ezy[x-1,y-1,currentTime-1])
            ezx[x-1,y-1,currentTime]=((epsilon-0.5*dt*sigmax[x,y])/(epsilon+0.5*dt*sigmax[x,y]))*ezx[x-1,y-1,currentTime-1]+((dt/dy)/(epsilon+0.5*dt*sigmax[x,y]))*(-hx[x,y,currentTime]+hx[x,y-1,currentTime])
            ezy[x-1,y-1,currentTime]=((epsilon-0.5*dt*sigmay[x,y])/(epsilon+0.5*dt*sigmay[x,y]))*ezy[x-1,y-1,currentTime-1]+((dt/dy)/(epsilon+0.5*dt*sigmay[x,y]))*(hy[x,y,currentTime]-hy[x-1,y,currentTime])
            #高斯激励源 参考《电磁波时域有限方法（第二版）》P.118
            ezx[(int)(sizeX/2),(int)(sizeY/2),currentTime] = math.exp(-(4 * constants.pi * math.pow((currentTime - shment), 2)) / math.pow(slen, 2))
            ezy[(int)(sizeX/2),(int)(sizeY/2),currentTime] = math.exp(-(4 * constants.pi * math.pow((currentTime - shment), 2)) / math.pow(slen, 2))
            #高斯激励源 参考《电磁波时域有限方法（第二版）》P.118
            ez[:,:,currentTime]=ezx[:,:,currentTime]+ezy[:,:,currentTime]

ez = numpy.zeros((sizeX, sizeY, maxTime))
hx = numpy.zeros((sizeX, sizeY, maxTime))
hy = numpy.zeros((sizeX, sizeY, maxTime))  # 初始化场量

for currentTime in range(1, maxTime):  # 在时间步长里迭代
    for x in range(boundWidth, sizeX - boundWidth):
        for y in range(boundWidth, sizeY - boundWidth):
            ez[x, y, currentTime] = ez[x, y, currentTime - 1] + (dt / epsilon) * (
                        (hy[x, y, currentTime - 1] - hy[x - 1, y, currentTime - 1]) / dx - (
                            hx[x, y, currentTime - 1] - hx[x, y - 1, currentTime - 1]) / dy)
            # 电场EZ分量递推公式 参考《电磁波时域有限方法（第二版）》P.23
            ez[(int)(sizeX / 2), (int)(sizeY / 2), currentTime] = math.exp(
                -(4 * constants.pi * math.pow((currentTime - shment), 2)) / math.pow(slen, 2))
            # 高斯激励源 参考《电磁波时域有限方法（第二版）》P.118
            # ez[(int)(sizeX/4),(int)(sizeY/4),currentTime] = math.exp(-(4 * constants.pi * math.pow((currentTime - shment+20), 2)) / math.pow(slen*2, 2))
            # 高斯激励源 参考《电磁波时域有限方法（第二版）》P.118

    for x in range(boundWidth, sizeX - boundWidth):
        for y in range(boundWidth, sizeY - boundWidth):
            hx[x, y, currentTime] = hx[x, y, currentTime - 1] - (dt / mu) * (
                        ez[x, y + 1, currentTime] - ez[x, y, currentTime]) / dy
            # 磁场Hx分量递推公式 参考《电磁波时域有限方法（第二版）》P.23
            hy[x, y, currentTime] = hy[x, y, currentTime - 1] + dt / mu * (
                        ez[x + 1, y, currentTime] - ez[x, y, currentTime]) / dx
            # 磁场Hy分量递推公式 参考《电磁波时域有限方法（第二版）》P.23


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
import matplotlib.animation as animation
fps = 30 # frame per sec
frn = maxTime # frame number of the animation
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x_plot = numpy.arange(0, sizeX, 1)
y_plot = numpy.arange(0, sizeY, 1)
x_plot, y_plot = numpy.meshgrid(x_plot, y_plot)
#surf = ax.plot_surface(x_plot, y_plot,ez[:,:,i], cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
ax.set_zlim(-1.0, 1.0)
plot = [ax.plot_surface(x_plot, y_plot,ez[:,:,0],
                       linewidth=0, antialiased=False)]
def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = surf = ax.plot_surface(x_plot, y_plot,ez[:,:,frame_number], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(ez[:,:,:], plot), interval=maxTime/fps)
fn = '2D BPML仿真'
ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)