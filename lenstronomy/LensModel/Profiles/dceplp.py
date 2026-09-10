__author__ = "pierrefleury"

import numpy as np

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["DCEPLP"]


class DCEPLP(LensProfileBase):
    """
    The drifting cored elliptical power-law potential (DCEPLP) is a lens model
    whose isopotential contours are ellipses with constant ellipticity and
    orientation, but whose centre drifts linearly with the elliptical radius.

    The radial profile of the potential is regular near the centre, and behaves
    as a power law far from it.
    
    Expressions to be written.
    """


    param_names = ["theta_E", "theta_core", "sharpness", "outer_slope",
                   "e1", "e2", "N2_drift", "phi_drift", "center_x", "center_y"]
    
    lower_limit_default = {
        "theta_E": 0,
        "theta_core": 1e-6,
        "sharpness": 1,
        "outer_slope": 0,
        "e1": -0.5,
        "e2": -0.5,
        "N2_drift": 0,
        "phi_drift": 0,
        "center_x": -100,
        "center_y": -100,
    }

    upper_limit_default = {
        "theta_E": 100,
        "theta_core": 10,
        "sharpness": 5,
        "outer_slope": 10,
        "e1": 0.5,
        "e2": 0.5,
        "N2_drift": 1,
        "phi_drift": 2*np.pi,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        super(DCEPLP, self).__init__()


    def psi_radial(self, theta, theta_E, theta_core, sharpness, outer_slope):
        """
        Axially symmetric cored power law potential.
        """
        
        power = (3 - outer_slope) / sharpness
        R = theta_core / theta_E
        
        # normalisation to ensure that the Einstein radius is theta_E
        psi0 = theta_E**2 / (3 - outer_slope) * R**sharpness * (1 + R**(-sharpness))**(1 - power)
        
        psi  = psi0 * (1 + (theta/theta_core)**sharpness)**power
        
        return psi


    def psi_prime(self, theta, theta_E, theta_core, sharpness, outer_slope):
       """
       First derivative of the radial potential (axysymmetric displacement angle).
       """

       power = (3 - outer_slope) / sharpness
       alpha = theta_E * (theta/theta_E)**(sharpness-1) * (
            (theta_core**sharpness + theta**sharpness)/(theta_core**sharpness + theta_E**sharpness))**(power - 1)

       return alpha


    def psi_second(self, theta, theta_E, theta_core, sharpness, outer_slope):
        """
        Second derivative of the radial potential.
        """

        alpha = self.psi_prime(theta, theta_E, theta_core, sharpness, outer_slope)
        
        alphaprime = (sharpness - 1 + (3 - outer_slope - sharpness)
                      * theta**sharpness / (theta_core**sharpness + theta**sharpness)) * alpha / theta
        
        return alphaprime


    def ellipse_matrix(self, e1, e2):
        """
        Returns the matrix involved in the equation of an ellipse
        in rectangular coordinates.
        """

        E_xx = (1 + e1**2 + e2**2)**0.5 - e1
        E_xy = - e2
        E_yx = - e2
        E_yy = (1 + e1**2 + e2**2)**0.5 + e1

        return E_xx, E_xy, E_yx, E_yy


    def EV(self, e1, e2, v_x, v_y):
        """
        Returns the vector E*V.
        """

        E_xx, E_xy, E_yx, E_yy = self.ellipse_matrix(e1, e2)

        EV_x = E_xx * v_x + E_xy * v_y
        EV_y = E_yx * v_x + E_yy * v_y
                    
        return EV_x, EV_y


    def VEW(self, e1, e2, v_x, v_y, w_x, w_y):
        """
        Returns the inner product V*E*W.
        """

        EW_x, EW_y = self.EV(e1, e2, w_x, w_y)
        VEW = v_x * EW_x + v_y * EW_y
                
        return VEW


    def drift_vector(self, e1, e2, N2_drift, phi_drift):
        """Computes the x and y component of the drift vector from its elliptical
        components:
        - squared elliptical radius, zeta2_drift
        - elliptical angle, phi_drift
        """
         
        # elliptical magnitude of the drift vector
        N_drift = N2_drift**0.5
                  
        # get (ellipse matrix)^-(1/2)
        e = (e1**2 + e2**2)**0.5
        q = (1 + e**2)**0.5 - e
        etilde1 = - q**0.5 / (1 + q) * e1
        etilde2 = - q**0.5 / (1 + q) * e2
                  
        drift_x, drift_y = self.EV(etilde1, etilde2, np.cos(2*phi_drift), np.sin(2*phi_drift))
        drift_x *= N_drift
        drift_y *= N_drift
                  
        return drift_x, drift_y


    def elliptical_radius(self, x, y, e1, e2, drift_x, drift_y):
        """
        Calculate the elliptical radius zeta of the ellipse containing (x, y),
        assuming constant ellipticity (e1, e2) and a linear drift of the centre
        with radius:
        x0 = drift_x * zeta
        y0 = drift_y * zeta
        """
        
        DED = self.VEW(e1, e2, drift_x, drift_y, drift_x, drift_y)
        DEX = self.VEW(e1, e2, drift_x, drift_y, x, y)
        XEX = self.VEW(e1, e2, x, y, x, y)
        
        zeta = (((1 - DED) * XEX + DEX**2)**0.5 - DEX) / (1 - DED)

        return zeta


    def elliptical_radius_gradient(self, x, y, e1, e2, drift_x, drift_y):
            """
            Gradient of the elliptical radius zeta_x, zeta_y.
            """

            zeta = self.elliptical_radius(x, y, e1, e2, drift_x, drift_y)

            x1 = x - drift_x * zeta
            y1 = y - drift_y * zeta

            x2, y2 = self.EV(e1, e2, x1, y1)
            DX2 = drift_x * x2 + drift_y * y2

            zeta_x = x2 / (zeta + DX2)
            zeta_y = y2 / (zeta + DX2)
    
            return zeta_x, zeta_y


    def elliptical_radius_hessian(self, x, y, e1, e2, drift_x, drift_y):
                """
                Hessian of the elliptical radius zeta_xx, zeta_xy, zeta_yx, zeta_yy.
                """

                zeta = self.elliptical_radius(x, y, e1, e2, drift_x, drift_y)
                zeta_x, zeta_y = self.elliptical_radius_gradient(x, y, e1, e2, drift_x, drift_y)

                E_xx, E_xy, E_yx, E_yy = self.ellipse_matrix(e1, e2)
        
                ED_x, ED_y = self.EV(e1, e2, drift_x, drift_y)
                XED        = x * ED_x + y * ED_y
                DED        = drift_x * ED_x + drift_y * ED_y

                denom = (1 + DED) * zeta - XED

                zeta_xx = (E_xx - ED_x * zeta_x - ED_x * zeta_x - (1 - DED) * zeta_x * zeta_x) / denom
                zeta_xy = (E_xy - ED_x * zeta_y - ED_y * zeta_x - (1 - DED) * zeta_x * zeta_y) / denom
                zeta_yx = zeta_xy
                zeta_yy = (E_yy - ED_y * zeta_y - ED_y * zeta_y - (1 - DED) * zeta_y * zeta_y) / denom

                return zeta_xx, zeta_xy, zeta_yx, zeta_yy


    def function(self, x, y, theta_E, theta_core, sharpness, outer_slope, e1, e2, N2_drift, phi_drift, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_core: core radius (in angles)
        :param sharpness: sharpness of the core-to-power-law transition
        :param outer_slope: power-law index far from the core
        :param e1: ellipticity component
        :param e2: ellipticity component
        :param drift_x: drift rate of the ellipse centre along x
        :param drift_y: drift rate of the ellipse centre along y
        :param center_x: x-position of lens centre (in angles)
        :param center_y: y-position of lens centre (in angles)
        :return: lensing potential (in squared angles)
        """

        x_ = x - center_x
        y_ = y - center_y

        # get drift vector from elliptical components
        drift_x, drift_y = self.drift_vector(e1, e2, N2_drift, phi_drift)

        # elliptical radius
        zeta = self.elliptical_radius(x, y, e1, e2, drift_x, drift_y)

        # potential
        psi  = self.psi_radial(zeta, theta_E, theta_core, sharpness, outer_slope)

        return psi


    def derivatives(self, x, y, theta_E, theta_core, sharpness, outer_slope, e1, e2, N2_drift, phi_drift, center_x=0, center_y=0):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_core: core radius (in angles)
        :param sharpness: sharpness of the core-to-power-law transition
        :param outer_slope: power-law index far from the core
        :param e1: ellipticity component
        :param e2: ellipticity component
        :param drift_x: drift rate of the ellipse centre along x
        :param drift_y: drift rate of the ellipse centre along y
        :param center_x: x-position of lens centre (in angles)
        :param center_y: y-position of lens centre (in angles)
        :return: displacement angle (in angles)
        """

        x_ = x - center_x
        y_ = y - center_y

        # get drift vector from elliptical components
        drift_x, drift_y = self.drift_vector(e1, e2, N2_drift, phi_drift)

        # elliptical radius and its gradient
        zeta = self.elliptical_radius(x, y, e1, e2, drift_x, drift_y)
        zeta_x, zeta_y = self.elliptical_radius_gradient(x, y, e1, e2, drift_x, drift_y)

        # radial derivative of the potential
        alpha = self.psi_prime(zeta, theta_E, theta_core, sharpness, outer_slope)

        # displacement angle
        alpha_x = alpha * zeta_x
        alpha_y = alpha * zeta_y

        return alpha_x, alpha_y


    def hessian(self, x, y, theta_E, theta_core, sharpness, outer_slope, e1, e2, N2_drift, phi_drift, center_x=0, center_y=0):
        """
        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :param theta_core: core radius (in angles)
        :param sharpness: sharpness of the core-to-power-law transition
        :param outer_slope: power-law index far from the core
        :param e1: ellipticity component
        :param e2: ellipticity component
        :param drift_x: drift rate of the ellipse centre along x
        :param drift_y: drift rate of the ellipse centre along y
        :param center_x: x-position of lens centre (in angles)
        :param center_y: y-position of lens centre (in angles)
        :return: Hessian matrix (dimensionless)
        """

        x_ = x - center_x
        y_ = y - center_y

        # get drift vector from elliptical components
        drift_x, drift_y = self.drift_vector(e1, e2, N2_drift, phi_drift)

        # elliptical radius and its derivatives
        zeta = self.elliptical_radius(x, y, e1, e2, drift_x, drift_y)
        zeta_x, zeta_y = self.elliptical_radius_gradient(x, y, e1, e2, drift_x, drift_y)
        zeta_xx, zeta_xy, zeta_yx, zeta_yy = self.elliptical_radius_hessian(x, y, e1, e2, drift_x, drift_y)

        # radial derivatives of the  potential
        alpha = self.psi_prime(zeta, theta_E, theta_core, sharpness, outer_slope)
        alpha_prime = self.psi_second(zeta, theta_E, theta_core, sharpness, outer_slope)

        # calculation of the Hessin
        H_xx = alpha_prime * zeta_x * zeta_x + alpha * zeta_xx
        H_xy = alpha_prime * zeta_x * zeta_y + alpha * zeta_xy
        H_yx = alpha_prime * zeta_y * zeta_x + alpha * zeta_yx
        H_yy = alpha_prime * zeta_y * zeta_y + alpha * zeta_yy

        return H_xx, H_xy, H_yx, H_yy