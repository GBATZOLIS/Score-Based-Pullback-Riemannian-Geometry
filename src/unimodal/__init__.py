class Unimodal:
    def __init__(self, diffeomorphism, strongly_convex) -> None:
        assert diffeomorphism.d == strongly_convex.d
        self.d = diffeomorphism.d
        self.phi = diffeomorphism # Diffeomorphism
        self.psi = strongly_convex # StronlyConvex
    

    def density(self, x):
        return self.psi.forward(self.phi.forward(x))
    
    def score(self, x): # TODO here we can just take the autograd of density
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def forward(self, x): 
        """ evaluate pseudo score """
        return self.psi.grad_forward(self.phi.forward(x))
    
    def differential_forward(self, x, X):
        """ evaluate differential of pseudo score """
        return self.psi.differential_grad_forward(self.phi.forward(x), self.phi.differential_forward(x, X))