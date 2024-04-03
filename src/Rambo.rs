use color_eyre::eyre::eyre;
use rand::Rng;

/// Convenience function to calculate the Minkowski scalar product of two four-vectors.
pub fn scalar(p: &[f64; 4], q: &[f64; 4]) -> f64 {
    return p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3];
}

fn newton(N: usize, s: f64, vecs: &[[f64; 4]], masses: &[f64]) -> f64 {
    let mut x: f64 = 0.5;
    let sqs = s.sqrt();
    let mut fx = - sqs;
    let mut iter = 0;
    let eps= f64::EPSILON;
    let mut xsq : f64;
    let mut fpx: f64;
    let mut psq: f64;
    let mut tmp: f64;

    while fx.abs() > eps && iter < 10000 {
        fx = - sqs;
        fpx = 0.0;
        xsq = x.powi(2);
        for i in 0..N {
            psq = vecs[i][0].powi(2);
            tmp = (masses[i].powi(2) + xsq * psq).sqrt();
            fx += tmp;
            fpx += psq / tmp;
        }
        fpx *= x;
        x -= fx / fpx;
        iter += 1;
    }
    return x;
}

/// Implementation of a RAMBO phase space generator based on the Fortran implementation of GoSam.
/// The number of momenta to generate is inferred from the number of supplied masses, the first
/// `n_in` masses and momenta are interpreted as incoming, the rest as outgoing.
///
/// Returns an error if `masses.len() < n_in + 1`.
pub fn rambo<R: Rng>(scale2: f64, masses: &Vec<f64>, n_in: usize, rng: &mut R) -> eyre::Result<Vec<[f64; 4]>> {
    if masses.len() <= n_in {
        return Err(eyre!("Too few masses in RAMBO: found {:?}, but at least n_in + 1 = {} masses are required.", masses, n_in + 1));
    }
    let mut vecs: Vec<[f64; 4]> = vec![[0.0; 4]; masses.len()];
    if n_in == 2 {
        let m1sq = masses[0].powi(2);
        let m2sq = masses[1].powi(2);
        let sqrts = 2.0 * scale2.sqrt();
        let A = (scale2 + m1sq - m2sq) / sqrts;
        let B = (scale2 + m2sq - m1sq) / sqrts;
        vecs[0][0] = A;
        vecs[0][3] = (A.powi(2) - m1sq).sqrt();
        vecs[1][0] = B;
        vecs[1][3] = -(B.powi(2) - m2sq).sqrt();
    } else {
        vecs[0][0] = masses[0];
    }
    let N = masses.len() - n_in;
    if N > 1 {
        let mut q: Vec<[f64; 4]> = Vec::with_capacity(N);
        let mut u: [f64; 4];
        for _ in 0..N {
            u = rng.gen();
            let sin_theta = 2.0 * (u[0] * (1.0 - u[0])).sqrt();
            let phi = 2.0 * std::f64::consts::PI * u[1];
            let E = - (u[2] * u[3]).ln();
            q.push([E, E * phi.cos() * sin_theta, E * phi.sin() * sin_theta, E * (2.0 * u[0] - 1.0)]);
        }
        let mut v: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
        for i in 0..=3 {
            v[i] = q.iter().map(|qi| qi[i]).sum();
        }
        let vsq = scalar(&v, &v);
        let invM = 1.0/vsq.sqrt();
        let b = [-invM * v[1], -invM * v[2], -invM * v[3]];
        let gamma = (1.0 + b[0].powi(2) + b[1].powi(2) + b[2].powi(2)).sqrt();
        let a = 1.0 / (1.0 + gamma);
        let x = invM * scale2.sqrt();
        let mut bq;
        for i in 0..N {
            bq = b[0] * q[i][1] + b[1] * q[i][2] + b[2] * q[i][3];
            vecs[n_in + i][0] = x * (gamma * q[i][0] + bq);
            vecs[n_in + i][1] = x * (b[0] * (q[i][0] + a * bq) + q[i][1]);
            vecs[n_in + i][2] = x * (b[1] * (q[i][0] + a * bq) + q[i][2]);
            vecs[n_in + i][3] = x * (b[2] * (q[i][0] + a * bq) + q[i][3]);
        }
        let x = newton(N, scale2, &vecs[n_in..], &masses[n_in..]);
        let xsq = x.powi(2);

        for i in n_in..masses.len() {
            vecs[i][0] = (masses[i].powi(2) + xsq * vecs[i][0].powi(2)).sqrt();
            vecs[i][1] *= x;
            vecs[i][2] *= x;
            vecs[i][3] *= x;
        }
    } else {
        for i in 0..3 {
            vecs[masses.len()][i] = vecs[0..masses.len()-1].iter().map(|v| v[i]).sum();
        }
    }

    return Ok(vecs);
}

/// Convenience function to boost N vectors into the center of mass frame in a 2 -> N process.
/// The first two momenta are interpreted as initial, the remaining N as final. The CMS energy is
/// inferred from the incoming momenta.
pub fn boost_to_cms(vecs: &mut Vec<[f64; 4]>) -> eyre::Result<()> {
    let s_cms = 2.0 * (vecs[0][0]*vecs[1][0] - vecs[0][1]*vecs[1][1] - vecs[0][2]*vecs[1][2] - vecs[0][3]*vecs[1][3]);
    return if s_cms < 0.0 {
        Err(eyre!("Encountered negative CMS energy while boosting {:?}", vecs))
    } else {
        let p_cms = [vecs[0][0] + vecs[1][0], vecs[0][1] + vecs[1][1], vecs[0][2] + vecs[1][2], vecs[0][3] + vecs[1][3]];
        let gamma = p_cms[0] / s_cms.sqrt();
        let mut beta = [0.0, 0.0, 0.0];
        for i in 0..vecs.len() {
            let mut bdopt = 0.0;
            for j in 1..3 {
                beta[j - 1] = p_cms[j] / p_cms[0];
                bdopt += vecs[i][j] * beta[j]
            }
            vecs[i][0] = gamma * (vecs[i][0] - bdopt);
            vecs[i][1] += gamma * beta[0] * (gamma / (1.0 + gamma) * bdopt - vecs[i][0]);
            vecs[i][2] += gamma * beta[1] * (gamma / (1.0 + gamma) * bdopt - vecs[i][0]);
            vecs[i][3] += gamma * beta[2] * (gamma / (1.0 + gamma) * bdopt - vecs[i][0]);
        }
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_pcg::Pcg64Mcg;
    const N_ITER_TEST: usize = 1_000;
    #[test]
    fn rambo_test() {
        let mut rng = Pcg64Mcg::seed_from_u64(187549175);
        for _ in 0..N_ITER_TEST {
            let masses = vec![0.0, 0.0, 125.0, 125.0, 0.0, 0.0];
            let mut vecs = rambo(500.0_f64.powi(2), &masses, 2, &mut rng).unwrap();
            boost_to_cms(&mut vecs).unwrap();
            let E_ref = 0.5 * vecs.iter().map(|p| p[0].abs()).sum::<f64>();
            for i in 1..=3 {
                let prec = vecs.iter().map(|p| p[i]).sum::<f64>().abs() / E_ref;
                if prec > 1E-7 {
                    println!("{i}: {} ({} digits)", prec, -prec.log10());
                }
                assert!(prec < 1E-7);
            }
        }
    }
}