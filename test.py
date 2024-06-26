import cellworld as cw
import cellworld_game as cg
import matplotlib.pyplot as plt

import cellworld_belief as belief
import torch
import pandas as pd
import numpy as np
import json
from cellworld_game import Visibility, CellWorldLoader
loader = CellWorldLoader("030_12_0063") #loader creates a bunch of objects about the world from plotly, pytorch, etc
visiblity = Visibility(arena=loader.arena, occlusions = loader.occlusions)
# w = loader.world
# vp = visiblity.get_visibility_polygon(src=(.05,.5), direction = 0, view_field=360)
# d = cw.Display(w)
# #grab polygon vetices for visible field and plot
# x = vp.vertices[:,0].cpu()
# y = vp.vertices[:,1].cpu()
# d.ax.plot(x,y)
base_path = ''
# experiment: cw.Experiment = cw.Experiment.load_from_file("/content/test_experiment.json")
f = 'PEEK03_20231205_1142_FMM19_030_12_0063_RT3.hdf'
exp_df = pd.read_hdf(base_path + f)

import math
model = cg.BotEvade(world_name="030_12_0063",
                real_time=False,
                render=False,
                use_predator=True)
model.predator.max_forward_speed = .30 / 90 / 2.34
model.prey.view_field = 270
DB = belief.DecreasingBeliefComponent(rate=.05)
NB = belief.NoBeliefComponent()
LOS = belief.LineOfSightComponent(other_scale=.5)
V = belief.VisibilityComponent()
D = belief.DiffusionComponent()
GD = belief.GaussianDiffusionComponent()
DD = belief.DirectedDiffusionComponent()
O = belief.OcclusionsComponent()
A = belief.ArenaComponent()
M = belief.MapComponent()
NL = belief.ProximityComponent()
components = [D, V,  LOS, M]
bs = belief.BeliefState(model=model,
                        definition=100,
                        components=components,
                        agent_name="prey",
                        other_name="predator",
                        probability=1)
def plot_pose(d,pose, scatter_size = None, color = None, alpha = 1):
  ears_x = []
  ears_y = []
  body_x = []
  body_y = []
  ear_ind = [1,2]
  body_ind = [0,3,4,5,6,7]
  for i in ear_ind:
    part = pose[i]
    if part['score'] > 0.8:
      ears_x.append(part['location']['x'])
      ears_y.append(part['location']['y'])
      d.ax.scatter(part['location']['x'], part['location']['y'], color = color, alpha = alpha, s = scatter_size)
  if len(ears_x) == len(ear_ind):
    d.ax.plot(ears_x,ears_y, color = color, alpha = alpha)
  for i in body_ind:
    part = pose[i]
    if part['score'] > 0.8:
      body_x.append(part['location']['x'])
      body_y.append(part['location']['y'])
      d.ax.scatter(part['location']['x'], part['location']['y'], color = color, alpha = alpha,  s = scatter_size)
  if len(body_x) > 1:
    for j, x in enumerate(body_x[1:]):
      d.ax.plot(body_x[j:j+2], body_y[j:j+2],color = color, alpha = alpha)
vis_prob_cleared = []
for e, row in exp_df.iloc[4:6].iterrows():
    print(e)
    episode = row['ep_data']
    model.reset()
    trajectories = episode.trajectories.split_by_agent()
    prey_trajectory: cw.Trajectories = trajectories["prey"]
    predator_trajectory: cw.Trajectories = trajectories["predator"]
    for s, step in enumerate(prey_trajectory[:]):
      frame = step.frame
      pose_data = json.loads(step.data) #THIS IS FOR LOADING POSE BY FRAME
      prey_rotation = step.rotation
      eye_loc = cw.Location(np.mean([pose_data[0]['location']['x'],pose_data[3]['location']['x']]),
                      np.mean([pose_data[0]['location']['y'],pose_data[3]['location']['y']]))
      agents_states = {}
      agents_states["prey"] = cg.AgentState(location=eye_loc.get_values(),
                                              direction=90 - prey_rotation)
      predator_step = predator_trajectory.get_step_by_frame(frame=step.frame)
      if predator_step:
        agents_states["predator"] = cg.AgentState(location=(predator_step.location.x, predator_step.location.y),
                                                    direction=90 - predator_step.rotation)
      # if not predator_step and s == 0:
      #   agent_state = model.agents['predator'].reset()
      #   agents_states["predator"]= agent_state
      model.set_agents_state(agents_state=agents_states)
      bs.tick()
      # model.step()
      bs_in_view = model.prey.visibility_polygon.contains(bs.points)
      in_view_matrix = torch.reshape(bs_in_view, bs.shape)
      #obtain summation of probabilities in visual field before trimming it for every step
      view_prob = bs.probability_distribution.clone()
      view_prob[torch.logical_not(in_view_matrix)] = 0
      # print(np.sum(view_prob.cpu().numpy()))
      vis_prob_cleared.append(np.sum(view_prob.cpu().numpy()))
      #update line_of_sight
      #step belief state
      #look at peeking with respsect to summation of probabilities
      #look at future trajectory and get probability of puff for future trajectory
      #look at line of sight to goal
      bs_prob = bs.probability_distribution.cpu().numpy()
      vp = bs.agent.visibility_polygon
      # vp = visiblity.get_visibility_polygon(src=eye_loc.get_values(), direction = 90 - prey_rotation, view_field=270)
      # if e == 4 and frame == prey_trajectory.get('frame')[-1]:
      #   d = cw.Display(loader.world, fig_size=(10,10), padding=0, cell_edge_color="lightgrey", background_color="white", habitat_edge_color="black")
      #   plot_pose(d,pose_data, scatter_size = 20, color = 'red', alpha = 1)
      #   d.arrow(eye_loc, theta = math.radians(prey_rotation), dist = 0.1, head_width = 0.01, color = 'green')
      #   x = np.concatenate([vp.vertices[:,0].cpu().numpy(),vp.vertices[0:1,0].cpu().numpy()])
      #   y = np.concatenate([vp.vertices[:,1].cpu().numpy(),vp.vertices[0:1,1].cpu().numpy()])
      #   d.ax.plot(x,y)
      #   d.ax.imshow(bs_prob, cmap='viridis', aspect='equal', origin = 'lower', extent=[0, 1, 0.07, 0.93],alpha = 0.3)
      if e==5 and frame == 0:
        d = cw.Display(loader.world, fig_size=(10,10), padding=0, cell_edge_color="lightgrey", background_color="white", habitat_edge_color="black")
        plot_pose(d,pose_data, scatter_size = 20, color = 'red', alpha = 1)
        d.arrow(eye_loc, theta = math.radians(prey_rotation), dist = 0.1, head_width = 0.01, color = 'green')
        x = np.concatenate([vp.vertices[:,0].cpu().numpy(),vp.vertices[0:1,0].cpu().numpy()])
        y = np.concatenate([vp.vertices[:,1].cpu().numpy(),vp.vertices[0:1,1].cpu().numpy()])
        d.ax.plot(x,y)
        d.ax.imshow(bs_prob, cmap='viridis', aspect='equal', origin = 'lower', extent=[0, 1, 0.07, 0.93],alpha = 0.3)
      if row.peek_list[frame] == 1:
        print(e)
        #Check if robot polygon intersects
        # print(vp.intersects(robot_polygon))
        d = cw.Display(loader.world, fig_size=(10,10), padding=0, cell_edge_color="lightgrey", background_color="white", habitat_edge_color="black")
        plot_pose(d,pose_data, scatter_size = 20, color = 'red', alpha = 1)
        d.arrow(eye_loc, theta = math.radians(prey_rotation), dist = 0.1, head_width = 0.01, color = 'green')
        x = np.concatenate([vp.vertices[:,0].cpu().numpy(),vp.vertices[0:1,0].cpu().numpy()])
        y = np.concatenate([vp.vertices[:,1].cpu().numpy(),vp.vertices[0:1,1].cpu().numpy()])
        d.ax.plot(x,y)
        d.ax.imshow(bs_prob, cmap='viridis', aspect='equal', origin = 'lower', extent=[0, 1, 0.07, 0.93],alpha = 0.3)
        #plot robot polygon
        rp = model.agents['predator'].get_body_polygon()
        #grap polgyon vertices for robot
        robot_x = np.concatenate([rp.vertices[:,0].cpu().numpy(),rp.vertices[0:1,0].cpu().numpy()])
        robot_y = np.concatenate([rp.vertices[:,1].cpu().numpy(),rp.vertices[0:1,1].cpu().numpy()])
        d.ax.plot(robot_x,robot_y)
        plt.show()
        plt.close()
        # d.ax.add_patch(rp)
        break