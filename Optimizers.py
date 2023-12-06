
class Momentum():

    def __init__(self, neuralnet, rate1, rate2):
        self.neuralnet = neuralnet
        self.param_layers = []
        self.moments = {}
        self.rate1 = rate1
        self.rate2 = rate2

        for l in self.neuralnet.layers:
            if hasattr(l, 'weights'):
                self.param_layers.append(l)
                self.moments[l] = None

        

    def step(self):
        for l in self.param_layers:
            #first step with no existing moment
            if(self.moments[l] is None):
                self.moments[l] = l.gradient.copy()
                for i in range(len(l.weights)):
                    l.weights[i] = l.weights[i] - self.rate1*l.gradient[i]
                    self.moments[l][i] = self.rate1*self.moments[l][i]
                

            #existing moment    
            else:
                for i in range(len(l.weights)):
                    l.weights[i] = l.weights[i] - self.rate1*l.gradient[i] - self.rate2*self.moments[l][i]
                    self.moments[l][i] = self.rate1*l.gradient[i] + self.rate2*self.moments[l][i]