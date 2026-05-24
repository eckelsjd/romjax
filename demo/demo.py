import romjax as romx

import jax
import equinox as eqx


class CustomRoutine(romx.Routine):

    init_params: romx.random.SamplerCallable

    def run(self):
        # t = np.linspace(0, 2*np.pi, 50)
        # def generate_sinusoid():
        #     for i in range(len(t)):
        #         yield t[:i+1], np.sin(t[:i+1])

        # x = np.linspace(0, 2*np.pi, 100)
        # sin = {"kind": "line", "data": (x, np.sin(x)), "name": "sin", "kwargs": { "c": "r", "ls": "-", "lw": 6 }}
        # cos = {"kind": "line", "data": (x, np.cos(x)), "name": "cos", "kwargs": { "c": "b", "ls": "--"}}
        # straight = {"kind": "line", "data": (x, x), "name": "straight", "kwargs": { "lw": 1 }}

        # moving = {"kind": "line", "data": list(generate_sinusoid()), 
        #           "opts": dict(xlabel="t", ylabel="y(t)", animate=True, ylim=(-1, 1))}

        # fig, ax, ani = romx.gridplot([(sin, cos, straight), moving], save="ani.mp4")
        # plt.show()

        sample = self.init_params(jax.random.key(0))
        print(sample)

        eqx.tree_pprint("Exiting custom routine")
        return 0
