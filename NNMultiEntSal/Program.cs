using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNMultiEntSal
{
    class Program
    {
        static void Main(string[] args)
        {
            
            double[,] ent = new double[2, 6] { { 5, 7, 8, -9, -5, 2 }, { -20, 60, -0, -10, 70, -40 } };
            double[,] sal = new double[2, 3] { { .7, .2, .5 }, { .8, .9, .1 } };
            

            //double[,] ent = new double[1, 6] { { 5, 7, 8, -9, -5, 2 } };
            //double[,] sal = new double[1, 5] { { .7, .2, .8, .9, .4 } };

            double apren = 1;
            int epocas = 10;

            ent = Trans(ent);
            sal = Trans(sal);

            int[] NeuNet = { ent.GetLength(0), 5, 4, sal.GetLength(0) };

            double[][,] pesos = IniPesos(NeuNet);
            double[][,] bias = IniSesgos(NeuNet);

            //Vemos como estan las salidas red neuronal 
            Console.WriteLine("**************************Red Neuronal Inicial**************************");
            (double[][,] sum, double[][,] act, double error) = EjecutarNN(ent, sal, pesos, bias, true);
            (double[,][,] sumM, double[,][,] actM) = MultiNNSumAct(pesos, bias, ent);

            //Entrenamiento de la red neuronal
            for (int k = 1; k <= 100; k++)
            {
                //(pesos, bias) = BackPopagation(NeuNet.Length, pesos, bias, sum, act, sal, apren);
                (pesos, bias) = BackPopagation(NeuNet.Length, pesos, bias, sumM, actM, sal, apren); ;
                (sum, act, error) = EjecutarNN(ent, sal, pesos, bias, false);
            }
            Console.WriteLine("**************************Red Neuronal Final**************************");

            //Vemos como quedaron las salidas de la red neuronal
            (sum, act, error) = EjecutarNN(ent, sal, pesos, bias, true);


            Console.WriteLine("Finalizado o<|:)|");
            Console.ReadLine();
        }
        #region Biblioteca

        public static void printM(double[,] matriz)
        {
            int ren = matriz.GetLength(0);
            int col = matriz.GetLength(1);
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
        public static void printM(double[][,] Arreglo)
        {
            Console.Write("\n");

            for (int j = 0; j < Arreglo.Length; j++)
            {
                Console.Write("-----------Matriz {0}-----------", j);
                printM(Arreglo[j]);
            }
            Console.Write("\n");
        }

        #region Operaciones de matrices
        /// <summary>
        /// Multiplica matrices de forma normal
        /// </summary>
        /// <param name="m1">Matriz izquierda</param>
        /// <param name="m2">Matriz derecha</param>
        /// <returns></returns>
        public static double[,] multi(double[,] m1, double[,] m2)
        {
            int ren = m1.GetLength(0);
            int col = m2.GetLength(1);
            double[,] m3 = new double[ren, col];
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
                m3[i, 0] = DCa[i, 0] * dAz[i, 0];
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
                    m3[i, j] = m1[i, j] + m2[i, j];
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
        #endregion

        #region Operaciones de red
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
            double sumerror = 0;
            for (int i = 0; i < a.GetLength(0); i++)
            {
                sumerror = sumerror + EC(a[i, 0], y[i, 0]);
            }
            return sumerror / a.Length;
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
        /// Crea una matriz identidad con el valor alfa el cual es el aprendizaje
        /// </summary>
        /// <param name="Dim"></param>
        /// <param name="alfa"></param>
        /// <returns></returns>
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
        /// <summary>
        /// Realiza la operacion de ajuste para los valores de pesos y sesgos
        /// </summary>
        /// <param name="oldW"></param>
        /// <param name="devw"></param>
        /// <param name="alfa"></param>
        /// <returns></returns>
        public static double[,] Newval(double[,] oldW, double[,] devw, double alfa)
        {
            int ren = oldW.GetLength(0);
            int col = oldW.GetLength(1);
            double[,] res = new double[ren, col];

            res = resta(oldW, multi(Apren(ren, alfa), devw));

            return res;
        }
        #endregion

        /// <summary>
        /// Inicia el arreglo de las matrices de los pesos al azar(0.0 - 0.9)
        /// </summary>
        /// <param name="NN"></param>
        /// <returns></returns>
        public static double[][,] IniPesos(int[] NN)
        {
            int tam = NN.Length - 1;
            double[][,] weights = new double[tam][,];
            double[,] Wi;

            for (int i = 0; i < tam; i++)
            {
                Wi = new double[NN[i + 1], NN[i]];

                for (int j = 0; j < NN[i + 1]; j++)
                {
                    for (int k = 0; k < NN[i]; k++)
                    {
                        var guid = Guid.NewGuid();
                        var justNumbers = new String(guid.ToString().Where(Char.IsDigit).ToArray());
                        var seed = int.Parse(justNumbers.Substring(0, 4));

                        var random = new Random(seed);
                        double value = Convert.ToDouble(random.Next(0, 9)) / 10;
                        Wi[j, k] = value;
                    }
                }
                weights[i] = Wi;
            }
            /*
            //Se debe corregir para que los inicialize al azar
            weights[0] = Pesos1();
            weights[1] = Pesos2();
            */
            return weights;
        }
        /// <summary>
        /// Inicia el arreglo de las matrices de los sesgos al azar(0.0 - 0.9)
        /// </summary>
        /// <param name="NN"></param>
        /// <returns></returns>
        public static double[][,] IniSesgos(int[] NN)
        {
            int tam = NN.Length - 1;
            double[][,] bias = new double[tam][,];
            double[,] bi;

            for (int i = 0; i < tam; i++)
            {
                bi = new double[NN[i + 1], 1];

                for (int j = 0; j < NN[i + 1]; j++)
                {
                    var guid = Guid.NewGuid();
                    var justNumbers = new String(guid.ToString().Where(Char.IsDigit).ToArray());
                    var seed = int.Parse(justNumbers.Substring(0, 4));

                    var random = new Random(seed);
                    double value = Convert.ToDouble(random.Next(0, 9)) / 10;
                    bi[j, 0] = value;
                }
                bias[i] = bi;
            }
            /*
            //Se debe corregir para que los inicialize al azar
            bias[0] = bias1();
            bias[1] = bias2();
            */
            return bias;
        }

        /// <summary>
        /// Obtiene las derivadas para los pesos y sesgos
        /// </summary>
        /// <param name="Capas"></param>
        /// <param name="MWeig"></param>
        /// <param name="Msum"></param>
        /// <param name="Mact"></param>
        /// <param name="Msal"></param>
        /// <returns></returns>
        public static (double[][,], double[][,]) BackErrores(int Capas, double[][,] MWeig, double[][,] Msum, double[][,] Mact, double[,] Msal)
        {
            Capas -= 1;
            double[][,] AjusteW = new double[Capas][,];
            double[][,] dif = new double[Capas][,];

            Capas -= 1;

            //double[,] dif1 = MDECac(Mact[Capas + 1], Msal);
            //double[,] dif2 = MDACsum(Msum[Capas]);


            dif[Capas] = multiAlg(MDACsum(Msum[Capas]), MDECac(Mact[Capas + 1], Msal));
            AjusteW[Capas] = multi(dif[Capas], Trans(Mact[Capas]));
            Capas -= 1;

            for (int L = Capas; L >= 0; L--)
            {
                dif[L] = difan(MDACsum(Msum[L]), MWeig[L + 1], dif[L + 1]);
                AjusteW[L] = DWa(dif[L], Mact[L]);
            }

            return (AjusteW, dif);
        }
        public static (double[][,], double[][,]) BackErrores(int longCap, double[][,] MWeig, double[,][,] Msum, double[,][,] Mact, double[,] Msal)
        {
            int Capas = longCap;
            Capas -= 1;
            int cant = Msal.GetLength(1);
            double[,][,] AjusteW = new double[cant,Capas][,];
            double[][,] AjusteWTo = new double[Capas][,];
            double[,][,] dif = new double[cant, Capas][,];
            double[][,] difTo = new double[Capas][,];

            Capas -= 1;


            for (int i = 0; i < cant; i++)
            {
                dif[i, Capas] = multiAlg(MDACsum(Msum[i, Capas]), MDECac(Mact[i, Capas + 1], Msal));
                AjusteW[i, Capas] = multi(dif[i, Capas], Trans(Mact[i, Capas]));
            }
            Capas -= 1;
            for (int i = 0; i < cant; i++)
            {
                for (int L = Capas; L >= 0; L--)
                {
                    dif[i, L] = difan(MDACsum(Msum[i, L]), MWeig[L + 1], dif[i, L + 1]);
                    AjusteW[i, L] = DWa(dif[i, L], Mact[i, L]);
                }
            }
            AjusteWTo = promedioDif(AjusteW);
            difTo = promedioDif(dif);
            Capas -= 2;
            /*AjusteW[Capas] = multi(difto[Capas], Trans(Mact[Capas]));
            Capas -= 1;

            for (int L = Capas; L >= 0; L--)
            {
                AjusteW[L] = DWa(dif[L], Mact[L]);
            }*/
            //////////////////////////////////////////////////////////////////////////


            return (AjusteWTo, difTo);
        }
        /// <summary>
        /// Con las derivadas de los pesos y sesgos da como resultados nuevos pesos y sesgos
        /// </summary>
        /// <param name="alfa"></param>
        /// <param name="MWeig"></param>
        /// <param name="Mbias"></param>
        /// <param name="DW"></param>
        /// <param name="Db"></param>
        /// <returns></returns>
        public static double[][,] promedioDif(double[,][,] dife)
        {
            int entr = dife.GetLength(0);
            int cant = dife.GetLength(1);
            double[][,] difProm = IniDif(cant, dife);
            double[][,] difProm2 = IniDif(cant, dife);

            for (int i = 0; i < cant; i++)
            {
                for (int j = 0; j < entr; j++)
                {
                    difProm[i] = suma(difProm[i], dife[j, i]);
                    //difProm[i]=dife[0]
                }
            }

            double total = 1.0 / Convert.ToDouble(entr);

            for (int j = 0; j < entr; j++)
            {
                double[,] m1 = Apren(difProm[j].GetLength(0), total);

                difProm2[j] = multi(m1, difProm[j]);
            }
            

            return difProm2;
        }
        public static double[][,] IniDif(int ca, double[,][,] difer)
        {
            double[][,] dif = new double[ca][,];
            int[] tam = new int[ca];

            for (int i = 0; i < ca; i++)
            {
                double[,] mat = difer[0,i];
                tam[i] = mat.GetLength(0);
            }

            for (int i = 0; i < ca; i++)
            {
                double[,] mat = new double[tam[i], 1];
                for (int j = 0; j < ca; j++)
                {
                    mat[j, 0] = 0;
                }
                dif[i] = mat;
            }
            return dif;
        }
        public static double[][,] IniPesos(int ca, double[,][,] difer)
        {
            double[][,] dif = new double[ca][,];
            int[] tam = new int[ca];

            for (int i = 0; i < ca; i++)
            {
                double[,] mat = difer[0, i];
                tam[i] = mat.GetLength(0);
            }

            for (int i = 0; i < ca; i++)
            {
                double[,] mat = new double[tam[i], 1];
                for (int j = 0; j < ca; j++)
                {
                    mat[j, 0] = 0;
                }
                dif[i] = mat;
            }
            return dif;
        }
        public static (double[][,], double[][,]) BackAjuste(double alfa, double[][,] MWeig, double[][,] Mbias, double[][,] DW, double[][,] Db)
        {
            int tam = MWeig.Length;
            double[][,] newMweig = new double[tam][,];
            double[][,] newMbias = new double[tam][,];
            for (int j = 0; j < tam; j++)
            {
                newMweig[j] = Newval(MWeig[j], DW[j], alfa);
                newMbias[j] = Newval(Mbias[j], Db[j], alfa);
            }

            return (newMweig, newMbias);
        }
        /// <summary>
        /// HAce el ajuste de pesos y sesgos
        /// </summary>
        /// <param name="Cap"></param>
        /// <param name="MWeig"></param>
        /// <param name="Mbias"></param>
        /// <param name="sum"></param>
        /// <param name="act"></param>
        /// <param name="Msal"></param>
        /// <param name="alfa"></param>
        /// <returns></returns>
        public static (double[][,], double[][,]) BackPopagation(int Cap, double[][,] MWeig, double[][,] Mbias, double[][,] sum, double[][,] act, double[,] Msal, double alfa)
        {
            int tam = MWeig.Length;
            double[][,] newMweig = new double[tam][,];
            double[][,] newMbias = new double[tam][,];

            (double[][,] devpesos, double[][,] devsesgos) = BackErrores(Cap, MWeig, sum, act, Msal);

            (newMweig, newMbias) = BackAjuste(alfa, MWeig, Mbias, devpesos, devsesgos);

            return (newMweig, newMbias);
        }
        
        public static (double[][,], double[][,]) BackPopagation(int Cap, double[][,] MWeig, double[][,] Mbias, double[,][,] sum, double[,][,] act, double[,] Msal, double alfa)
        {
            int tam = MWeig.Length;
            int cant = Msal.GetLength(1);
            double[][,] newMweig = new double[tam][,];
            double[][,] newMbias = new double[tam][,];

            //double[,][,] totMweig = new double[,tam][,];
            //double[,][,] totMbias = new double[,tam][,];

            double[][,] devpesos; double[][,] devsesgos;

            (devpesos, devsesgos) = BackErrores(Cap, MWeig, sum, act, Msal);

            //(newMweig, newMbias) = BackAjuste(alfa, MWeig, Mbias, devpesos, devsesgos);

            return (newMweig, newMbias);
        }
        /// <summary>
        /// Ejecuta la red neuronal con los pesos, sesgos y entradas que tengan 
        /// </summary>
        /// <param name="Wei"></param>
        /// <param name="Bia"></param>
        /// <param name="entra"></param>
        /// <returns></returns>
        public static (double[][,], double[][,]) NNSumAct(double[][,] Wei, double[][,] Bia, double[,] entra)
        {
            int cont = Wei.Length;
            double[][,] act = new double[cont + 1][,];
            double[][,] sum = new double[cont][,];

            act[0] = entra;

            for (int j = 0; j < cont; j++)
            {
                sum[j] = suma(multi(Wei[j], act[j]), Bia[j]);
                act[j + 1] = actsigm(sum[j]);
            }

            return (sum, act);
        }
        public static (double[,][,], double[,][,]) MultiNNSumAct(double[][,] Wei, double[][,] Bia, double[,] entra)
        {
            int cont = Wei.Length;
            int num = entra.GetLength(1);

            double[,][,] act = new double[num, cont + 1][,];
            double[,][,] sum = new double[num, cont][,];

            for (int k = 0; k < num; k++)
            {
                act[k, 0] = ObtenerCol(entra, k);
            }


            for (int j = 0; j < num; j++)
            {
                for (int k = 0; k < cont; k++)
                {
                    sum[j, k] = suma(multi(Wei[k], act[j, k]), Bia[k]);
                    act[j, k + 1] = actsigm(sum[j, k]);
                }
            }

            return (sum, act);
        }
        /// <summary>
        /// Ejecuta la red neuronal para darnos los arrgelos de matrices de sumas y activaciones ya que se usaran para entrenarlas
        /// Tambien da el error de la red neuronal actual
        /// </summary>
        /// <param name="entr"></param>
        /// <param name="sali"></param>
        /// <param name="Mweig"></param>
        /// <param name="Mbias"></param>
        /// <param name="print"></param>
        /// <returns></returns>
        public static (double[][,], double[][,], double) EjecutarNN(double[,] entr, double[,] sali, double[][,] Mweig, double[][,] Mbias, bool print)
        {
            int ren = sali.GetLength(0);
            int col = sali.GetLength(1);
            double[,] res = new double[ren, col];

            (double[][,] sum, double[][,] act) = NNSumAct(Mweig, Mbias, entr);

            double Error = sumEC(act[Mweig.Length], sali);
            if (print)
            {
                Console.WriteLine("//////////////////Salida de red//////////////////");
                printM(act[Mweig.Length]);
                Console.WriteLine("//////////////////Valor esperado//////////////////");
                printM(sali);
                Console.WriteLine("//////////////////Error//////////////////");
                Console.WriteLine(Error);
            }

            return (sum, act, Error);
        }
        public static double[,] ObtenerCol(double[,] entr,int col)
        {
            int ele = entr.GetLength(0);
            double[,] res = new double[ele, 1];
            double valor;

            for (int k = 0; k < ele; k++)
            {
                valor = entr[k, col];
                res[k, 0] = valor;
            }

            return res;
        }
        /*public static (double[,][,], double[,][,], double) MEjecutarNN(double[,] entr, double[,] sali, double[][,] Mweig, double[][,] Mbias, bool print)
        {
            int ren = sali.GetLength(0);
            int col = sali.GetLength(1);
            int num = entr.GetLength(1);

            double[,] res = new double[ren, col];

            for (int i = 0; i < num; i++)
            {

            }

            (double[,][,] sum, double[,][,] act) = NNSumAct(Mweig, Mbias, entr);

            double Error = sumEC(act[Mweig.Length], sali);
            if (print)
            {
                Console.WriteLine("//////////////////Salida de red//////////////////");
                printM(act[Mweig.Length]);
                Console.WriteLine("//////////////////Valor esperado//////////////////");
                printM(sali);
                Console.WriteLine("//////////////////Error//////////////////");
                Console.WriteLine(Error);
            }

            return (sum, act, Error);
        }*/
    }
    #endregion
}

