# Source: https://github.com/thomasvnl/rd-to-wgs84/blob/master/conversions.py

# MIT License

# Copyright (c) 2019 Thomas Verschoof

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
The formulas in this class were based on a white paper by ing. F.H.
Schreutelkamp from "Stichting De Koepel" and ir. G.L. Strang van Hees, former
scholar at TU Delft.

Unfortunately, as of January 1st 2014, the foundation "De Koepel" has halted
all their activities and suspended their website. As the original article can
no longer be found on the website, I've made sure to host a backup of it at:
http://media.thomasv.nl/2015/07/Transformatieformules.pdf (in Dutch). Please
consult this white paper for the origin of the coefficients used below.

I take no credit for the formulas used, all I have done was convert the
formulas into an easy to work with Python class for usage in other projects.
All credit for the formulas go to F.H. Schreutelkamp and G.L. Strang van Hees.
"""
from builtins import enumerate


class RDWGS84Converter(object):

    x0 = 155000
    y0 = 463000
    phi0 = 52.15517440
    lam0 = 5.38720621

    # Coefficients or the conversion from RD to WGS84
    Kp = [0, 2, 0, 2, 0, 2, 1, 4, 2, 4, 1]
    Kq = [1, 0, 2, 1, 3, 2, 0, 0, 3, 1, 1]
    Kpq = [3235.65389, -32.58297, -0.24750, -0.84978, -0.06550, -0.01709,
           -0.00738, 0.00530, -0.00039, 0.00033, -0.00012]

    Lp = [1, 1, 1, 3, 1, 3, 0, 3, 1, 0, 2, 5]
    Lq = [0, 1, 2, 0, 3, 1, 1, 2, 4, 2, 0, 0]
    Lpq = [5260.52916, 105.94684, 2.45656, -0.81885, 0.05594, -0.05607,
           0.01199, -0.00256, 0.00128, 0.00022, -0.00022, 0.00026]
    # Coefficients for the conversion from WGS84 to RD
    Rp = [0, 1, 2, 0, 1, 3, 1, 0, 2]
    Rq = [1, 1, 1, 3, 0, 1, 3, 2, 3]
    Rpq = [190094.945, -11832.228, -114.221, -32.391, -0.705, -2.340,
           -0.608, -0.008, 0.148]

    Sp = [1, 0, 2, 1, 3, 0, 2, 1, 0, 1]
    Sq = [0, 2, 0, 2, 0, 1, 2, 1, 4, 4]
    Spq = [309056.544, 3638.893, 73.077, -157.984, 59.788, 0.433, -6.439,
           -0.032, 0.092, -0.054]

    def from_rd(self, x: int, y: int) -> tuple:
        """
        Converts RD coordinates into WGS84 coordinates
        """
        dx = 1E-5 * (x - self.x0)
        dy = 1E-5 * (y - self.y0)
        latitude = self.phi0 + sum([v * dx ** self.Kp[i] * dy ** self.Kq[i]
                                    for i, v in enumerate(self.Kpq)]) / 3600
        longitude = self.lam0 + sum([v * dx ** self.Lp[i] * dy ** self.Lq[i]
                                     for i, v in enumerate(self.Lpq)]) / 3600

        return latitude, longitude

    def from_wgs84(self, latitude: float, longitude: float) -> tuple:
        """
        Converts WGS84 coordinates into RD coordinates
        """
        dlat = 0.36 * (latitude - self.phi0)
        dlon = 0.36 * (longitude - self.lam0)
        x = self.x0 + sum([v * dlat ** self.Rp[i] * dlon ** self.Rq[i]
                           for i, v in enumerate(self.Rpq)])
        y = self.y0 + sum([v * dlat ** self.Sp[i] * dlon ** self.Sq[i]
                           for i, v in enumerate(self.Spq)])

        return x, y
