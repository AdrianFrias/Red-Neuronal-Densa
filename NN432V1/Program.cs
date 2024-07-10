using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN432V1
{
    class Program
    {
        static void Main(string[] args)
        {
            double[,] ent = new double[4, 1] { { 5 }, { 6 }, { 7 }, { 8 } };
            double[,] sal = new double[2, 1] { { .7 }, { .2 } };
            double alf = 1;

            double[,] w1 = new double[3, 4];
            double[,] b1 = new double[3, 1];
            double[,] w2 = new double[2, 3];
            double[,] b2 = new double[2, 1];

            double[][,] prueba = new double[4][,];

            w1 = Pesos1();
            b1 = bias1();
            w2 = Pesos2();
            b2 = bias2();

            prueba[0] = ent;
            prueba[1] = w1;
            prueba[2] = b1;
            prueba[3] = sal;


            printM(ent);

            Console.WriteLine("Primera parte");
            Console.WriteLine("Pesos 1");
            printM(w1);
            Console.WriteLine("sesgos 1");
            printM(b1);
            double[,] sum1 = suma(multi(w1, ent), b1);
            Console.WriteLine("Suma 1");
            printM(sum1);
            Console.WriteLine("Activacion 1");
            double[,] act1 = actsigm(sum1);
            printM(act1);

            Console.WriteLine("Segunda parte");
            Console.WriteLine("Pesos 2");
            printM(w2);
            Console.WriteLine("Sesgos 2");
            printM(b2);
            Console.WriteLine("Suma 2");
            double[,] sum2 = suma(multi(w2, act1), b2);
            printM(sum2);
            Console.WriteLine("Activacion 2");
            double[,] act2 = actsigm(sum2);
            printM(act2);

            Console.WriteLine("ERROR");
            printM(MEC(act2, sal));
            Console.WriteLine("\n" + sumEC(act2, sal));

            Console.WriteLine("--------------Backprpagation CAPA L---------------");
            Console.WriteLine("Derivada de coste respecto a la activacion");
            double[,] DError = MDECac(act2, sal);
            printM(DError);

            Console.WriteLine("Derivada de la activacion respecto a la suma");
            double[,] DActiv = MDACsum(sum2);
            printM(DActiv);

            Console.WriteLine("Diferencial 2");
            double[,] dif2 = multiAlg(DActiv, DError);
            printM(dif2);
            Console.WriteLine("Prueba de transposicion");
            printM(w1);
            printM(Trans(w1));

            Console.WriteLine("Derivada de la suma respecto a los pesos(w)");
            double[,] DSumW = Trans(act1);
            printM(DSumW);

            Console.WriteLine("Derivada del coste respecto a los pesos(w)");
            double[,] DW2 = multi(dif2, DSumW);
            printM(DW2);

            Console.WriteLine("Derivada del coste respecto a los sesgos(b)");
            double[,] DB2 = dif2;
            printM(DB2);

            Console.WriteLine("--------------Backpropagation CAPA L-1---------------");

            Console.WriteLine("Diferencial 1");
            double[,] dif1 = difan(MDACsum(sum1), w2, dif2);
            printM(dif1);

            Console.WriteLine("Derivada del coste respecto a los sesgos(b)");
            double[,] DB1 = dif1;
            printM(DB1);

            Console.WriteLine("Matriz de diferencial de pesos 1");
            double[,] DW11 = DWa(MDACsum(sum1), w2, dif2, ent);
            printM(DW11);

            Console.WriteLine("Matriz de diferencial de pesos 1");
            double[,] DW12 = DWa(dif1, ent);
            printM(DW12);

            Console.WriteLine("--------------Resulatdos de la Red en automatico---------------");
            double[,] res = EjecutarNN(ent, sal, w1, b1, w2, b2);
            printM(res);
            printM(act2);

            Console.WriteLine("--------------Actualizacion de datos---------------");
            Console.WriteLine("Nuevos pesos 2");
            double[,] W2new = New(w2, DW2, alf);
            printM(W2new);
            Console.WriteLine("Nuevos pesos 2");
            double[,] b2new = New(b2, DB2, alf);
            printM(b2new);
            Console.WriteLine("Nuevos pesos 1");
            double[,] W1new = New(w1, DW12, alf);
            printM(W1new);
            Console.WriteLine("Nuevos pesos 1");
            double[,] b1new = New(b1, DB1, alf);
            printM(b1new);

            Console.WriteLine("--------------Resulatdos de la nueva Red en automatico---------------");
            double[,] res2 = EjecutarNN(ent, sal, W1new, b1new, W2new, b2new);
            printM(res2);
            Console.WriteLine("ERROR");
            printM(MEC(res2, sal));
            Console.WriteLine("\n" + sumEC(res2, sal));

            Console.WriteLine("\nFinalizado o<|:)|");
            Console.ReadLine();
        }

        #region Biblioteca

        public static void printM(double[,] matriz)
        {
            int ren=matriz.GetLength(0);
            int col=matriz.GetLength(1);
            Console.Write("\n");

            for (int j = 0; j < ren; j++)
            {
                Console.Write("|");
                for (int k = 0; k < col; k++)
                {
                    Console.Write("{0}|", matriz[j, k]);
                }
                Console.Write("\n");

            }
            Console.Write("\n");
        }
        public static void printM(double[] matriz)
        {
            int ren = matriz.GetLength(0);
            Console.Write("\n");

            for (int j = 0; j < ren; j++)
            {
                Console.Write("|");
                Console.Write("{0}|", matriz[j]);
                Console.Write("\n");
            }
        }
        
        #region Inicializar
        public static double[,] Pesos1()
        {
            double[,] Mpeso = new double[3, 4];
            Mpeso[0, 0] = 0.1;
            Mpeso[0, 1] = .4;
            Mpeso[0, 2] = -.3;
            Mpeso[0, 3] = .2;

            Mpeso[1, 0] = .1;
            Mpeso[1, 1] = -.4;
            Mpeso[1, 2] = 0.5;
            Mpeso[1, 3] = 0.6;

            Mpeso[2, 0] = 0.6;
            Mpeso[2, 1] = -0.9;
            Mpeso[2, 2] = .8;
            Mpeso[2, 3] = -.7;

            return Mpeso;
        }
        public static double[,] bias1()
        {
            double[,] Mpeso = new double[3, 1];
            Mpeso[0, 0] = 0.5;
            Mpeso[1, 0] = .4;
            Mpeso[2, 0] = .3;

            return Mpeso;
        }

        public static double[,] Pesos2()
        {
            double[,] Mpeso = new double[2, 3];
            Mpeso[0, 0] = 0.1;
            Mpeso[0, 1] = -.4;
            Mpeso[0, 2] = .7;

            Mpeso[1, 0] = -.1;
            Mpeso[1, 1] = .4;
            Mpeso[1, 2] = -0.5;

            return Mpeso;
        }
        public static double[,] bias2()
        {
            double[,] Mpeso = new double[2, 1];
            Mpeso[0, 0] = 0.5;
            Mpeso[1, 0] = .4;

            return Mpeso;
        }
        #endregion
        /// <summary>
        /// Multiplica matrices de forma normal
        /// </summary>
        /// <param name="m1">Matriz izquierda</param>
        /// <param name="m2">Matriz derecha</param>
        /// <returns></returns>
        public static double[,] multi (double[,] m1, double[,] m2)
        {
            int ren = m1.GetLength(0);
            int col = m2.GetLength(1);
            double[,] m3 = new double[ren,col];
            for (byte i = 0; i < ren; i++)
            {
                for (byte j = 0; j < col; j++)
                {
                    m3[i, j] = 0;
                    for (byte k = 0; k < m2.GetLength(0); k++)
                    {
                        m3[i, j] = m3[i, j] + (m1[i, k] * m2[k, j]);
                    }
                }
            }
            return m3;
        }
        /// <summary>
        /// Multplica matrices como si fueran numeros normales
        /// </summary>
        /// <param name="DCa"></param>
        /// <param name="dAz"></param>
        /// <returns></returns>
        public static double[,] multiAlg(double[,] DCa, double[,] dAz)
        {
            int ren = DCa.GetLength(0);
            int col = DCa.GetLength(1);
            double[,] m3 = new double[ren, col];
            for (byte i = 0; i < ren; i++)
            {
                m3[i,0]= DCa[i, 0] * dAz[i, 0];
            }

            return m3;

        }
        /// <summary>
        /// Suma 2 matrices
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static double[,] suma(double[,] m1, double[,] m2)
        {
            int ren = m1.GetLength(0);
            int col = m2.GetLength(1);
            double[,] m3 = new double[ren, col];
            for (byte i = 0; i < ren; i++)
            {
                for (byte j = 0; j < col; j++)
                {
                    m3[i, j] = m1[i, j]+ m2[i, j];
                }
            }
            return m3;
        }
        /// <summary>
        /// Resta dos matrices
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static double[,] resta(double[,] m1, double[,] m2)
        {
            int ren = m1.GetLength(0);
            int col = m2.GetLength(1);
            double[,] m3 = new double[ren, col];
            for (byte i = 0; i < ren; i++)
            {
                for (byte j = 0; j < col; j++)
                {
                    m3[i, j] = m1[i, j] - m2[i, j];
                }
            }
            return m3;
        }
        /// <summary>
        /// Transpone una matriz
        /// </summary>
        /// <param name="m1"></param>
        /// <returns></returns>
        public static double[,] Trans(double[,] m1)
        {
            int ren = m1.GetLength(1);
            int col = m1.GetLength(0);
            double[,] m3 = new double[ren, col];
            for (byte i = 0; i < ren; i++)
            {
                for (byte j = 0; j < col; j++)
                {
                    m3[i, j] = m1[j, i];
                }
            }
            return m3;
        }

        /// <summary>
        /// Fusncion de activacion sigmoide
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double sigm(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }
        /// <summary>
        /// Se forma una matriz de activaciones sigmoide
        /// </summary>
        /// <param name="m1"></param>
        /// <returns></returns>
        public static double[,] actsigm(double[,] m1)
        {
            int ren = m1.GetLength(0);
            int col = m1.GetLength(1);
            double[,] m3 = new double[ren, col];
            for (byte i = 0; i < m1.GetLength(0); i++)
            {
                for (byte j = 0; j < m1.GetLength(1); j++)
                {
                    m3[i, j] = sigm(m1[i, j]);
                }
            }
            return m3;
        }
        /// <summary>
        /// Error cuadratico entre la salida obtenida y requerida
        /// </summary>
        /// <param name="a"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static double EC(double a, double y)
        {
            return Math.Pow(a - y, 2);
        }
        /// <summary>
        /// Matriz del error cuadratico
        /// </summary>
        /// <param name="a"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static double[,] MEC(double[,] a, double[,] y)
        {
            int ren = a.GetLength(0);
            int col = a.GetLength(1);
            double[,] m3 = new double[ren, col];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                m3[i, 0] = EC(a[i, 0], y[i, 0]);
            }
            return m3;
        }
        /// <summary>
        /// Suma los errores cuadraticos de la matriz, no obtiene el promedio
        /// </summary>
        /// <param name="a"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static double sumEC(double[,] a, double[,] y)
        {
            double sumerror=0;
            for(int i = 0; i < a.GetLength(0); i++)
            {
                sumerror = sumerror + EC(a[i, 0], y[i, 0]);
            }
            return sumerror;
        }
        /// <summary>
        /// Derivada de la funcion de coste respecto a la activacion
        /// </summary>
        /// <param name="a"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static double DECac(double a, double y)
        {
            return 2 * (a - y);
        }
        /// <summary>
        /// Matriz de las Derivadas de la funcion de coste respecto a las activaciones
        /// </summary>
        /// <param name="a"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static double[,] MDECac(double[,] a, double[,] y)
        {
            int ren = a.GetLength(0);
            int col = a.GetLength(1);
            double[,] m3 = new double[ren, col];
            for (int i = 0; i < ren; i++)
            {
                m3[i, 0] = DECac(a[i, 0], y[i, 0]);
            }

            return m3;
        }
        /// <summary>
        /// Derivada de la funcion de activacion respecto a la suma
        /// </summary>
        /// <param name="sum"></param>
        /// <returns></returns>
        public static double DACsum(double sum)
        {
            return sigm(sum) * (1 - sigm(sum));
        }
        /// <summary>
        /// Matriz de las Derivadas de la funcion de activacion respecto a las sumas
        /// </summary>
        /// <param name="sum"></param>
        /// <returns></returns>
        public static double[,] MDACsum(double[,] sum)
        {
            int ren = sum.GetLength(0);
            int col = sum.GetLength(1);
            double[,] m3 = new double[ren, col];
            for (int i = 0; i < ren; i++)
            {
                m3[i, 0] = DACsum(sum[i, 0]);
            }

            return m3;
        }
        /// <summary>
        /// Diferencial de las anteriores a L
        /// </summary>
        /// <param name="DACsum"></param>
        /// <param name="WL"></param>
        /// <param name="difL"></param>
        /// <returns></returns>
        public static double[,] difan(double[,] DACsum, double[,] WL, double[,] difL)
        {
            int ren = DACsum.GetLength(0);
            int col = DACsum.GetLength(1);
            double[,] DW = new double[ren, col];

            DW = multiAlg(DACsum, multi(Trans(WL), difL));

            return DW;
        }
        /// <summary>
        /// Matriz de valores obtenidos de las derivadas de pesos(w)
        /// </summary>
        /// <param name="DACsum">Derivada de activacion respecto a sumas L-1</param>
        /// <param name="WL">Matriz de pesos L</param>
        /// <param name="difL">Matriz de diferencial de L</param>
        /// <param name="act">Matriz de activacion L-1</param>
        /// <returns></returns>
        public static double[,] DWa(double[,] DACsum, double[,] WL, double[,] difL, double[,] act)
        {
            int ren = DACsum.GetLength(0);
            int col = act.GetLength(1);
            double[,] DW = new double[ren, col];

            DW = multi(multiAlg(DACsum, multi(Trans(WL), difL)), Trans(act));

            return DW;
        }
        /// <summary>
        /// Matriz de valores obtenidos de las derivadas de pesos(w)
        /// </summary>
        /// <param name="Difa">Matriz de diferencial de L-1</param>
        /// <param name="act">Matriz de activacion L-1</param>
        /// <returns></returns>
        public static double[,] DWa(double[,] Difa, double[,] act)
        {
            int ren = Difa.GetLength(0);
            int col = act.GetLength(1);
            double[,] DW = new double[ren, col];

            DW = multi(Difa, Trans(act));

            return DW;
        }
        /// <summary>
        /// Corre la neurona para este ejemplo
        /// </summary>
        /// <param name="entr"></param>
        /// <param name="sali"></param>
        /// <param name="pe1"></param>
        /// <param name="bi1"></param>
        /// <param name="pe2"></param>
        /// <param name="bi2"></param>
        /// <returns></returns>
        public static double[,] EjecutarNN(double[,] entr, double[,] sali, double[,] pe1, double[,] bi1, double[,] pe2, double[,] bi2)
        {
            int ren = sali.GetLength(0);
            int col = sali.GetLength(1);
            double[,] res = new double[ren, col];

            double[,] sum1 = suma(multi(pe1, entr), bi1);
            double[,] act1 = actsigm(sum1);

            double[,] sum2 = suma(multi(pe2, act1), bi2);
            double[,] act2 = actsigm(sum2);

            return act2;
        }
        public static double[,] Apren(int Dim, double alfa)
        {
            double[,] res = new double[Dim, Dim];

            for (byte i = 0; i < Dim; i++)
            {
                for (byte j = 0; j < Dim; j++)
                {
                    if (i == j)
                    {
                        res[i, j] = alfa;
                    }
                    else
                    {
                        res[i, j] = 0;
                    }
                }
            }

            return res;

        }

        public static double[,] New(double[,] oldW, double[,] devw, double alfa)
        {
            int ren = oldW.GetLength(0);
            int col = oldW.GetLength(1);
            double[,] res = new double[ren, col];

            res = resta(oldW, multi(Apren(ren, alfa), devw));

            return res;
        }

        #endregion
    }
}
