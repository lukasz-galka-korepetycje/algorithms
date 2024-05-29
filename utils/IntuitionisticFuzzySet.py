class IntuitionisticFuzzySet:
    def __init__(self, degreeOfMembership, degreeOfNonMembership, delta=0.001):
        if degreeOfMembership + delta < 0.0 or degreeOfMembership - delta > 1.0:
            raise ValueError("Degree of membership must be in [0.0;1.0].")
        if degreeOfNonMembership + delta < 0.0 or degreeOfNonMembership - delta > 1.0:
            raise ValueError("Degree of non-membership must be in [0.0;1.0].")
        if degreeOfMembership + degreeOfNonMembership - delta > 1.0:
            raise ValueError("Sum of degree of membership and degree of non-membership must be <= 1.0.")

        self.degreeOfMembership = degreeOfMembership
        self.degreeOfNonMembership = degreeOfNonMembership
        self.degreeOfUncertainty = 1.0 - degreeOfMembership - degreeOfNonMembership

    def __str__(self):
        return "Degree of membership = " + str(self.degreeOfMembership) + "\n" + "Degree of non-membership = " + str(
            self.degreeOfNonMembership) + "\n" + "Degree of uncertainty = " + str(self.degreeOfUncertainty)
