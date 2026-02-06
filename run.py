import time
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

model = PPO.load("tmp/best_model.zip")

# Hur lång paus per env.step (sekunder)
RENDER_DELAY = 1 / 60  # prova 1/30 om du vill ännu långsammare

def main():
    env = retro.make(game="SuperMarioBros-Nes")
    env = MaxAndSkipEnv(env, 4)

    observation = env.reset()
    done = False

    while not done:
        action, state = model.predict(observation)
        observation, reward, done, info = env.step(action)
        env.render()

        time.sleep(RENDER_DELAY)

        if done:
            observation = env.reset()
            done = True

if __name__ == "__main__":
    main()