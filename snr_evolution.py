# -*- coding: utf-8 -*-
"""

estimate SNR parameters from sizes and spectral fits

@author: Patrick Kavanagh (DIAS)

"""
from __future__ import print_function

import os
from configobj import ConfigObj
import optparse
import astropy.units as u
import astropy.constants as c


class SupernovaRemnant:
    """
    Class that loads supernova remnant properties determined from
    size/spectral fitting and derives various parameters such as age,
    etc.

    Input:
    snr_ini_file  -   the SNR parameter file in config object format


    Examples:
    ----
    To load the fitted parameters and determine all other values and print the age
    in years:

    > my_snr = SupernovaRemnant('my_snr_config.ini')
    > my_snr.load_all()
    > print(my_snr.age(units='yr'))

    ----
    To load everything and produce plots:

    > my_snr = SupernovaRemnant('my_snr_config.ini')
    > my_snr.load_all(plots=True)

    ----
    To run from command line and print all results to file:

    $ python snr_evolution.py my_snr_config.ini

    """
    def __init__(self, snr_ini_file):

        assert os.path.isfile(snr_ini_file), 'SNR parameter file not found!'
        self.properties = ConfigObj(snr_ini_file)

        # load the snr parameters to the SupernovaRemnant object
        self.name = self.properties['Name']['snr_name']
        self.alt_name = self.properties['Name']['alt_name']
        self.distance = float(self.properties['Name']['distance']) * u.kpc
        #
        self.size_major = float(self.properties['Size']['major_axis']) * u.pc
        self.size_minor = float(self.properties['Size']['minor_axis']) * u.pc
        self.size_error = float(self.properties['Size']['size_error']) * u.pc
        self.position_angle = float(self.properties['Size']['pos_angle']) * u.deg
        #
        self.shell_detection = self.properties['xspec_fits']['shell']['detection']
        self.shell_model = self.properties['xspec_fits']['shell']['model']['model']
        #
        self.shell_Nh = float(self.properties['xspec_fits']['shell']['absorption']['Nh']) * 1e22 / u.cm**2
        self.shell_Nh_l = float(self.properties['xspec_fits']['shell']['absorption']['Nh_l']) * 1e22 / u.cm**2
        self.shell_Nh_u = float(self.properties['xspec_fits']['shell']['absorption']['Nh_u']) * 1e22 / u.cm**2
        #
        self.shell_kT = float(self.properties['xspec_fits']['shell']['temperature']['kT']) * u.keV
        self.shell_kT_l = float(self.properties['xspec_fits']['shell']['temperature']['kT_l']) * u.keV
        self.shell_kT_u = float(self.properties['xspec_fits']['shell']['temperature']['kT_u']) * u.keV
        #
        self.shell_tau = float(self.properties['xspec_fits']['shell']['ionisation_parameter']['tau']) * u.s / u.cm**3
        self.shell_tau_l = float(self.properties['xspec_fits']['shell']['ionisation_parameter']['tau_l']) * u.s / u.cm**3
        self.shell_tau_u = float(self.properties['xspec_fits']['shell']['ionisation_parameter']['tau_u']) * u.s / u.cm**3
        #
        self.shell_norm = float(self.properties['xspec_fits']['shell']['normalisation']['norm']) * u.cm**-5
        self.shell_norm_l = float(self.properties['xspec_fits']['shell']['normalisation']['norm_l']) * u.cm**-5
        self.shell_norm_u = float(self.properties['xspec_fits']['shell']['normalisation']['norm_u']) * u.cm**-5
        #
        self.interior_detection = self.properties['xspec_fits']['Fe_interior']['detection']
        self.interior_model = self.properties['xspec_fits']['Fe_interior']['model']['model']
        #
        self.interior_kT = float(self.properties['xspec_fits']['Fe_interior']['temperature']['kT']) * u.keV
        self.interior_kT_l = float(self.properties['xspec_fits']['Fe_interior']['temperature']['kT_l']) * u.keV
        self.interior_kT_u = float(self.properties['xspec_fits']['Fe_interior']['temperature']['kT_u']) * u.keV
        #
        self.interior_abund = float(self.properties['xspec_fits']['Fe_interior']['abundance']['abund'])
        #
        self.interior_norm = float(self.properties['xspec_fits']['Fe_interior']['normalisation']['norm']) * u.cm ** 5
        self.interior_norm_l = float(self.properties['xspec_fits']['Fe_interior']['normalisation']['norm_l']) * u.cm ** 5
        self.interior_norm_u = float(self.properties['xspec_fits']['Fe_interior']['normalisation']['norm_u']) * u.cm ** 5

    def _calc_shock_speed(self, kT):
        """
        calculate the shock speed from the best fit temperature kT:

        v = (16 kT / 3 mu)**0.5

        where mu is the mean mass per particle = 0.61 m_p
        """
        mu = 0.61 * c.m_p
        T_K = kT.to(u.K, equivalencies=u.temperature_energy())

        # do calculation
        v_s = ((16 * c.k_B * T_K) / (3 * mu))**0.5

        # redefine units to be in m/s
        v_s = v_s.value * u.m / u.s

        # convert to km/s
        v_s = v_s.to(u.km/u.s)

        return v_s

    def _calc_volume_reff(self, major, minor, error):
        """
        calculate the volume and effective radius using the major and minor
        axes from the morphological fits and assuming that the third semi-principle
        axis is between semi-major and semi-minor

        returns:
        V_l     -   the lower limit on the volume
        V_u     -   the upper limit on the volume
        r_eff_l -   the lower limit on the effective radius
        r_eff_u -   the upper limit on the effective radius
        """
        # get the semi-axis in cm
        semi_major = (major.to(u.cm) + error.to(u.cm)) / 2.
        semi_minor = (minor.to(u.cm) - error.to(u.cm)) / 2.

        # determine the volume limits
        V_l = (4. / 3.) * 3.14159 * semi_major * semi_minor ** 2
        V_u = (4. / 3.) * 3.14159 * semi_major ** 2 * semi_minor

        # determine the effective area limits
        r_eff_l = ((3 * V_l) / (4 * 3.14159)) ** (1. / 3)
        r_eff_u = ((3 * V_u) / (4 * 3.14159)) ** (1. / 3)

        return V_l, V_u, r_eff_l, r_eff_u

    def _calc_age(self, v_s, R):
        """
        calculate the age from the similarity solution:

        v = 2R/5t --> t = 2R/5v

        where R is the effective radius
        """
        age = (2 * R.to(u.cm)) / (5 * v_s.to(u.cm/u.s))

        return age

    def _calc_emission_measure(self, K):
        """
        calculate the emission measure from the model normalisation and
        the formula:

        EM = K*4*pi*D^2 / 10^-14
        """
        EM = (K * 4 * 3.14159 * self.distance.to(u.cm)**2) / 1.e-14

        return EM

    def _calc_preshock_density(self, EM, V):
        """
        calculate the pre-shock density n_0 using the Sedov emission integral:

        EM =  2.07 (n_e/n_H) n_0,H**2 V

        where n_e / n_H = 1.21 and n_0 ~ 1.1 n_0,H
        """
        # get pre-shock H density
        n_0_H = (EM / (2.07 * 1.21 * V)) ** 0.5

        # get pre-shock density
        n_0 = 1.1 * n_0_H

        return n_0

    def _calc_explosion_energy(self, t, n_0, R):
        """
        calulate the explosion energy using the similarity solution:

        R = ((2.02 E_0 t**2) / mu_n n_0)**1/5

        where mu_n = 1.4 m_p

        """
        # calculate explosion energy
        E_0 = (R.to(u.cm)**5 * 1.4 * c.m_p * n_0) / (2.02 * t.to(u.s)**2)

        # convert to erg
        E_0 = E_0.to(u.erg)

        return E_0

    def _calc_swept_up_mass(self, V, n_0):
        """
        calculate the swept up mass using:

        M = V mu_n n_0

        where mu_n = 1.4 m_p
        """
        # calculate the mass
        M = V * 1.4 * c.m_p * n_0

        # convert to solar masses
        M = M.to(u.Msun)

        return M

    def _print_latex_table_row(self):
        """
        print row of latex table
        """
        # reformat units for some
        age_l = self.age_l.to(u.kyr)
        age_u = self.age_u.to(u.kyr)

        print('Name & $n_{0} & $v_{s}$ & $t$ & $M$ & $E_{0}$ \\')
        print('%s & %0.1e--%0.1e & %0.0f--%0.0f & %0.0f--%0.0f & %0.0f--%0.0f & %0.2e--%0.2e \\' % (self.name,
                                                                                                 self.n_0_l.value,
                                                                                                 self.n_0_u.value,
                                                                                                 self.v_s_l.value,
                                                                                                 self.v_s_u.value,
                                                                                                 age_l.value,
                                                                                                 age_u.value,
                                                                                                 self.su_mass_l.value,
                                                                                                 self.su_mass_u.value,
                                                                                                 self.E_0_l.value,
                                                                                                 self.E_0_u.value))

    def load_all(self):
        """
        run all the calculations, saving results to the SupernovaRemnant object
        """
        # shock speeds
        self.v_s = self._calc_shock_speed(self.shell_kT)
        self.v_s_l = self._calc_shock_speed(self.shell_kT_l)
        self.v_s_u = self._calc_shock_speed(self.shell_kT_u)

        # volume and effective radius
        self.V_l, self.V_u, self.r_eff_l, self.r_eff_u = self._calc_volume_reff(self.size_major, self.size_minor, self.size_error)

        # age
        self.age_l = self._calc_age(self.v_s_u, self.r_eff_l)
        self.age_u = self._calc_age(self.v_s_l, self.r_eff_u)

        # emission measure
        self.EM = self._calc_emission_measure(self.shell_norm)
        self.EM_l = self._calc_emission_measure(self.shell_norm_l)
        self.EM_u = self._calc_emission_measure(self.shell_norm_u)

        # pre-shock density
        self.n_0_l = self._calc_preshock_density(self.EM_l, self.V_u)
        self.n_0_u = self._calc_preshock_density(self.EM_u, self.V_l)

        # explosion energy
        self.E_0_l = self._calc_explosion_energy(self.age_u, self.n_0_l, self.r_eff_l)
        self.E_0_u = self._calc_explosion_energy(self.age_l, self.n_0_u, self.r_eff_u)

        # swept-up mass
        self.su_mass_l = self._calc_swept_up_mass(self.V_l, self.n_0_l)
        self.su_mass_u = self._calc_swept_up_mass(self.V_u, self.n_0_u)

        # print the latex row
        self._print_latex_table_row()

    def _test(self):
        """
        run some test lines
        """
        print(self.v_s, self.v_s_l, self.v_s_u)
        print(self.V_l, self.V_u)
        print(self.r_eff_l.to(u.pc), self.r_eff_u.to(u.pc))
        print(self.age_l.to(u.yr), self.age_u.to(u.yr))
        print(self.shell_norm, self.EM)
        print(self.n_0_l, self.n_0_u)
        print(self.E_0_l, self.E_0_u)
        print(self.su_mass_l, self.su_mass_u)



if __name__ == "__main__":
    """
    Determine SNR parameters from those derived in spectral and morphological fits.
    """
    # parse arguments
    usage = "Usage: %prog <snr_config_file> [options]"

    parser = optparse.OptionParser(usage)
    parser.add_option('-t', '--temp', dest='temp', action='store',
                      help="placeholder for options in future", default='temp')
    (options, args) = parser.parse_args()

    try:
        input_file = args[0]

        my_snr = SupernovaRemnant(input_file)
        #my_snr.load_all()

    except IndexError as e:
        print("IndexError: {0}".format(e))
        print("No input file provided!\n")
        print(parser.print_help())

    # run some test lines
    my_snr.load_all()
