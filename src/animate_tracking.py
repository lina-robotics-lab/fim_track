import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from matplotlib import animation
from matplotlib import style

style.use('fivethirtyeight')

filepath = '/home/tianpeng/track_log_data.pkl'

log = pkl.load(open(filepath,'rb'))

	
def plot_trajectory(ax,traj,name,marker):
	def direction_at(i):
		direction = traj[i+1,:]-traj[i,:]
		return traj[i,0],traj[i,1], direction[0],direction[1]
	
	ax.scatter(traj[:,0],traj[:,1],label=name,marker=marker)
	

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.set_xlim((0,10))
ax.set_ylim((0,10))

ax.set_aspect('equal')


# animation function. This is called sequentially
def animate(i):
	ax.clear()
	
	ax.set_xlim((0,10))
	ax.set_ylim((0,10))

	ax.set_aspect('equal')

	for key,val in log['est_locs_log'].items():
		if key in["multi_lateration","intersection","ekf","pf",'actual_loc']:
			ind = min([i,len(val)-1])
			plot_trajectory(ax,val[ind-1:ind],key,'.')

	for key,val in log['target_locs'].items():
		ind = min([i,len(val)-1])
		plot_trajectory(ax,val[ind-1:ind],key,'*')
    
	for key,val in log['sensor_locs'].items():
		ind = min([i,len(val)-1])
		plot_trajectory(ax,val[max([ind-4,0]):ind],key,'.')
		# plot_trajectory(ax,val,key,'*')
	
		# print(val[-1])
	
	for key,val in log['waypoints'].items():
		ind = min([i,len(val)-1])
		# print(len(val),key)
		plot_trajectory(ax,val[ind],"waypoints for {}".format(key),'+')
	
	# ax.set_title('Hill Climbing+ Estimation & FIM in Real Life\nFrame {}'.format(i))
	ax.set_title('Frame {}'.format(i))
	ax.legend(loc='upper left',bbox_to_anchor=(1, .8))




if not 'target_locs' in log.keys(): 
	log['target_locs'] = {}
	t = len(log['sensor_locs'][list(log['sensor_locs'].keys())[0]])
	log['target_locs']['target_0']=np.array([[2.4,4.5] for i in range(t)])


frames = np.inf
for key,val in log['target_locs'].items():
	frames = int(np.min([len(val),frames]))	

# call the animator. blit=True means only re-draw the parts that have changed.
# But if blit=True, the animate() function must return the list of artists.
# If blit = False, the animate() function does not have to return anything. Plus, redrawing the who canvas is cheap. 
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=500, blit=False)
# Change the (x,y) value in bbox_to_anchor argument to change the position of the legend box
plt.show()