import random
import cellworld_gym as cg
import cellworld_belief as belief

env = cg.BotEvadeEnv(world_name="21_05",
                     real_time=False,
                     render=True,
                     use_lppos=False,
                     use_predator=True)

DB = belief.DecreasingBeliefComponent(rate=.05)
NB = belief.NoBeliefComponent()
LOS = belief.LineOfSightComponent()
V = belief.VisibilityComponent()
D = belief.DiffusionComponent()
GD = belief.GaussianDiffusionComponent()
DD = belief.DirectedDiffusionComponent()
O = belief.OcclusionsComponent()
A = belief.ArenaComponent()
M = belief.MapComponent()
NL = belief.ProximityComponent()
#
# components = []
# if condition == 0:
#     components = [NL, M]
# if condition == 1:
#     components = [NB, LOS, M]
# elif condition == 2:
#     components = [V, LOS, M]
# elif condition == 3:
#     components = [GD, V, LOS, M]
# elif condition == 4:
#     components = [DD, V, LOS, M]
# elif condition == 5:
#     components = [DD, GD, V, LOS, M]

components = [DD, GD, V, LOS, M]


bs = belief.BeliefState(model=env.model,
                        definition=100,
                        components=components,
                        agent_name="prey",
                        other_name="predator",
                        probability=1)

action = 0
for i in range(100):
    obs, _ = env.reset()
    finished, truncated = False, False
    puff_count = 0
    while not finished and not truncated:
        prey_location = (obs.prey_x, obs.prey_y)
        if not env.model.paused:
            bs.tick()
            obs, reward, finished, truncated, info = env.step(action=action)
        else:
            action = random.randint(0, 220)
            env.model.step()
