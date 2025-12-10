
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_60cf532e5150c6417bb22c67ff3ae244 : public Expression
  {
     public:
       

       dolfin_expression_60cf532e5150c6417bb22c67ff3ae244()
       {
            _value_shape.push_back(2);
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = (({ A | A_{i_{15}, i_{16}} = -1 * ({ A | A_{i_{13}, i_{14}} = I[i_{13}, i_{14}] * f_9 * f_49 })[i_{15}, i_{16}] }) + ({ A | A_{i_9, i_{10}} = (sym(grad(f_48)))[i_9, i_{10}] * 2.0 * f_6 }) + ({ A | A_{i_{11}, i_{12}} = I[i_{11}, i_{12}] * f_7 * (tr(sym(grad(f_48)))) }))[0, 0];
          values[1] = (({ A | A_{i_{15}, i_{16}} = -1 * ({ A | A_{i_{13}, i_{14}} = I[i_{13}, i_{14}] * f_9 * f_49 })[i_{15}, i_{16}] }) + ({ A | A_{i_9, i_{10}} = (sym(grad(f_48)))[i_9, i_{10}] * 2.0 * f_6 }) + ({ A | A_{i_{11}, i_{12}} = I[i_{11}, i_{12}] * f_7 * (tr(sym(grad(f_48)))) }))[0, 1];

       }

       void set_property(std::string name, double _value) override
       {

       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {

       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_60cf532e5150c6417bb22c67ff3ae244()
{
  return new dolfin::dolfin_expression_60cf532e5150c6417bb22c67ff3ae244;
}

