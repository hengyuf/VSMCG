# VSMCG
玩原神玩的✧✧✧
In:
            for __ in range(num_steps_hidden):
                likelihood = self.log_total(weights, epsilon)
                # print("Likelihood:",likelihood)
                likelihood.backward()
                self.upd_param(lr=1e-5)
parameters are updated but eps are not. parameters overflow.
