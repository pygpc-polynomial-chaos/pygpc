#include <stddef.h>

int gpc(double* polynomial_coeffs, double* x, 
	double* gpc_matrix, size_t n_psi,
	size_t n_var, size_t n_x)
{	
	for (size_t i_xi = 0; i_xi < n_x; i_xi++) {
		//reset poly_coeffs counter
		size_t poly_coeffs_offset = 0;
		//iterate over all polynomials
		for(size_t i_psi = 0; i_psi < n_psi; i_psi++)
		{	
			double psi_value = 1;
			//iterate over all variables
			for(size_t var = 0; var < n_var; var++)
			{	
				//get degree of polynomial
				size_t polynom_degree = (size_t) polynomial_coeffs[poly_coeffs_offset];
				//get xi
				double xi = x[i_xi * n_var + var];
				//increment offset to get to first coefficient
				poly_coeffs_offset++;
				//get first polynomial coefficient (order zero)
				double polynom = polynomial_coeffs[poly_coeffs_offset];
				//loop over polynom coefficients of order greater zero
				for(size_t coeff = 1; coeff < polynom_degree + 1; coeff++)
				{	
					//increment offset to get to next coefficient
					poly_coeffs_offset++;
					//initialize fist power of xi (xi^1)
					double xi_power = xi;
					//compute power of xi
					for(size_t exponent = 1; exponent < coeff; exponent++)
					{
						xi_power *= xi;
					}
					//accumulate polynomial
					polynom += polynomial_coeffs[poly_coeffs_offset] * xi_power;
				}
				//increment offset to get to the order of the next polynomial
				poly_coeffs_offset++;
				//multiply value with next polynomial
				psi_value *= polynom;
			}
			//write resulting matrix
			gpc_matrix[i_xi * n_psi + i_psi] = psi_value;
		}
	}

	return 0;
}