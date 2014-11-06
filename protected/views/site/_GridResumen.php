<fieldset style="width:100%; margin: 15px 0;border: 1px solid #AC8B3A;">

    <table>

        <tr>

            <td>
                <?php
// compone la cadena de consulta
                $consultaSQL = '
    
SELECT COUNT(*) AS Expr1 FROM motores WHERE (Proceso ="'
                        . $proceso .
                        '")
';
// se prepara el comando de SQL como en http://www.yiiframework.com/doc/guide/1.1/en/database.dao
                $command = Yii::app()->db->createCommand($consultaSQL);
// se ejecuta la consulta y los resultados quedan en un arreglo de resultados $resultados[0] es el primero
                $resultados = $command->queryAll();
// imprime todo el arreglo de resultados
//print_r($resultados);
// ejemplo de uso de un campo de uno de los resultados (Equipo del resultado 0)
                //       echo "<br/>";
//        echo '<b>Expr1='.$resultados[0]['Expr1'].'</b>'
//echo 'hola='.$resultados[0]['Expr1'];
                ?>
                <table>
                    <tr>
                        <td>
                            <b>Total Motores:</b> <?php echo $resultados[0]['Expr1']; ?>
                        </td>
                    </tr>
                </table>

            </td>

            <td>
<?php
// compone la cadena de consulta
$consultaSQL = '
SELECT FORMAT( SUM(kW),2) AS Expr1 FROM motores WHERE (Proceso ="'
        . $proceso .
        '")
';
// se prepara el comando de SQL como en http://www.yiiframework.com/doc/guide/1.1/en/database.dao
$command = Yii::app()->db->createCommand($consultaSQL);
// se ejecuta la consulta y los resultados quedan en un arreglo de resultados $resultados[0] es el primero
$resultados = $command->queryAll();
// imprime todo el arreglo de resultados
//print_r($resultados);
// ejemplo de uso de un campo de uno de los resultados (Equipo del resultado 0)
//       echo "<br/>";
//        echo '<b>Expr1='.$resultados[0]['Expr1'].'</b>'
//echo 'hola='.$resultados[0]['Expr1'];
?>
                <table>
                    <tr>
                        <td>
                            <b>Potencia Instalada:</b> <?php echo $resultados[0]['Expr1']; ?>
                        </td>
                    </tr>
                </table>

            </td> 

            <td>

<?php
// compone la cadena de consulta
$consultaSQL = '
   SELECT COUNT(Toma) AS Expr1 FROM  
(
	SELECT Consulta1.TAG, Consulta2.Motor, Consulta2.kW, Consulta1.Expr1,Consulta2.VibLL, Consulta2.VibLA, Consulta2.Proceso, Consulta2.Temperatura, Consulta2.Toma
	FROM 
	(
	        SELECT 	Vibraciones_1.TAG, Max(Vibraciones_1.Fecha) AS Expr1
		FROM vibraciones AS Vibraciones_1
		GROUP BY Vibraciones_1.TAG
	)
	as Consulta1 INNER JOIN 
	(
	 	SELECT motores.TAG, motores.kW, vibraciones.VibLL, vibraciones.VibLA,
	 	vibraciones.Fecha, vibraciones.Toma, motores.Proceso,vibraciones.Temperatura, 
	 	motores.Motor
		FROM motores INNER JOIN vibraciones ON motores.TAG = vibraciones.TAG 
	)
	as Consulta2 ON (Consulta1.Expr1 = Consulta2.Fecha) AND (Consulta1.TAG = Consulta2.TAG)
)
as Consulta3 WHERE (Proceso ="'
        . $proceso .
        '")

';
// se prepara el comando de SQL como en http://www.yiiframework.com/doc/guide/1.1/en/database.dao
$command = Yii::app()->db->createCommand($consultaSQL);
// se ejecuta la consulta y los resultados quedan en un arreglo de resultados $resultados[0] es el primero
$resultados = $command->queryAll();
// imprime todo el arreglo de resultados
//print_r($resultados);
// ejemplo de uso de un campo de uno de los resultados (Equipo del resultado 0)
//       echo "<br/>";
//        echo '<b>Expr1='.$resultados[0]['Expr1'].'</b>'
//echo 'hola='.$resultados[0]['Expr1'];
?>
                <table>
                    <tr>
                        <td>
                            <b>Motores Analizados:</b> <?php echo $resultados[0]['Expr1']; ?>
                        </td>
                    </tr>
                </table>
                <!--TODO:falta arreglar numero de paginas para que no se vean supe rpuestas --->
            </td>

            <td>
<?php
// compone la cadena de consulta
$consultaSQL = '
    
SELECT COUNT(Toma) AS Expr1 FROM 
(
	SELECT Consulta3.Toma, Consulta3.Proceso, Consulta3.TAG, Consulta3.Motor, Consulta3.kW, Consulta3.Expr1, Consulta3.VibLA, Consulta3.VibLL, Consulta3.Temperatura
	FROM 
	(
		SELECT Consulta1.TAG, Consulta2.Motor, Consulta2.kW, Consulta1.Expr1, Consulta2.VibLL, Consulta2.VibLA, Consulta2.Proceso, Consulta2.Temperatura, Consulta2.Toma
                FROM 
                (
                SELECT Vibraciones_1.TAG, Max(Vibraciones_1.Fecha) AS Expr1
		FROM vibraciones AS Vibraciones_1
		GROUP BY Vibraciones_1.TAG
                
               	)
                as Consulta1 INNER JOIN 
                (
                SELECT motores.TAG, motores.kW, vibraciones.VibLL, vibraciones.VibLA, vibraciones.Fecha, vibraciones.Toma, motores.Proceso, vibraciones.Temperatura, motores.Motor
		FROM motores INNER JOIN vibraciones ON motores.TAG = vibraciones.TAG
                
                ) 
                as Consulta2 ON (Consulta1.Expr1 = Consulta2.Fecha) AND (Consulta1.TAG = Consulta2.TAG)
	)
	as Consulta3 WHERE (((Consulta3.kW)<15) AND ((Consulta3.VibLA)>2.8 And (Consulta3.VibLA)<4.5)) OR (((Consulta3.kW)<15) AND ((Consulta3.VibLA)>=4.5)) OR (((Consulta3.kW)<15) AND ((Consulta3.VibLL)>2.8 And (Consulta3.VibLL)<4.5)) OR (((Consulta3.kW)<15) AND ((Consulta3.VibLL)>=4.5)) OR (((Consulta3.kW)<75 And (Consulta3.kW)>=15) AND ((Consulta3.VibLA)>=2.8 And (Consulta3.VibLA)<7.1)) OR (((Consulta3.kW)<75 And (Consulta3.kW)>=15) AND ((Consulta3.VibLA)>=7.1)) OR (((Consulta3.kW)<75 And (Consulta3.kW)>=15) AND ((Consulta3.VibLL)>=2.8 And (Consulta3.VibLL)<7.1)) OR (((Consulta3.kW)<75 And (Consulta3.kW)>=15) AND ((Consulta3.VibLL)>=7.1)) OR (((Consulta3.kW)>=75) AND ((Consulta3.VibLA)>4.5 And (Consulta3.VibLA)<11)) OR (((Consulta3.kW)>=75) AND ((Consulta3.VibLA)>=11)) OR (((Consulta3.kW)>=75) AND ((Consulta3.VibLL)>4.5 And (Consulta3.VibLL)<11)) OR (((Consulta3.kW)>=75) AND ((Consulta3.VibLL)>=11))
)



as Riesgo WHERE (Proceso ="'
        . $proceso .
        '")
';
// se prepara el comando de SQL como en http://www.yiiframework.com/doc/guide/1.1/en/database.dao
$command = Yii::app()->db->createCommand($consultaSQL);
// se ejecuta la consulta y los resultados quedan en un arreglo de resultados $resultados[0] es el primero
$resultados = $command->queryAll();
// imprime todo el arreglo de resultados
//print_r($resultados);
// ejemplo de uso de un campo de uno de los resultados (Equipo del resultado 0)
//       echo "<br/>";
//        echo '<b>Expr1='.$resultados[0]['Expr1'].'</b>'
//echo 'hola='.$resultados[0]['Expr1'];
?>
                <table>
                    <tr>
                        <td>
                            <b>Motores en Riesgo:</b> <?php echo $resultados[0]['Expr1']; ?>
                        </td>
                    </tr>
                </table>

            </td>  

        </tr>
    </table>
</fieldset>




<div id="gridMotores" name="gridMotores">
<?php
$sql = '
SELECT TAG, Motor, kW, Expr1, VibLA, VibLL, Temperatura FROM

(
	SELECT Consulta1.TAG, Consulta2.Motor, Consulta2.kW, Consulta1.Expr1, Consulta2.VibLL, Consulta2.VibLA, Consulta2.Proceso, Consulta2.Temperatura, Consulta2.Toma
	FROM
	
	(
	        SELECT Vibraciones_1.TAG, Max(Vibraciones_1.Fecha) AS Expr1
		FROM vibraciones AS Vibraciones_1
		GROUP BY Vibraciones_1.TAG

	)
	as Consulta1 INNER JOIN
	
	(
		SELECT motores.TAG, motores.kW, vibraciones.VibLL, vibraciones.VibLA, vibraciones.Fecha, vibraciones.Toma, motores.Proceso, vibraciones.Temperatura, motores.Motor
		FROM motores INNER JOIN vibraciones ON motores.TAG = vibraciones.TAG

	)
	
	
	as Consulta2 ON (Consulta1.Expr1 = Consulta2.Fecha) AND (Consulta1.TAG = Consulta2.TAG)



)

as Riesgo WHERE (Proceso = "'.$proceso.'")  
';
$dataProvider = new CSqlDataProvider($sql, array(
            'id'=>'procesoDP',
            'keyField'=>'TAG',
            'pagination' => array(
                'pageSize' => 5,
            ),
        ));
$susana='Susy';
// Dibuja el Widget de gridview
$this->widget('zii.widgets.grid.CGridView', array(
    'id' => 'motores_grid',
    // TODO: Implementar incluir en searchequipos también el proceso.
    'dataProvider' => $dataProvider,
    //'dataProvider' => $model->search(),
    'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
    'columns' => array(
        'TAG',
        'Motor',
        'kWl',
       'Expr1::Fecha',

        'VibLA',
        'VibLL',
        'Temperatura',
    ),
));
?>
</div>


<!--

    <?php
    /*
      $oDbConnection = Yii::app()->db; // Getting database connection (config/main.php has to set up database
      // Here you will use your complex sql query using a string or other yii ways to create your query
      $oCommand = $oDbConnection->createCommand('SELECT TAG, Motor, kW, Expr1, VibLA, VibLL, Temperatura FROM

      (
      SELECT Consulta3.Toma, Consulta3.Proceso, Consulta3.TAG, Consulta3.Motor, Consulta3.kW, Consulta3.Expr1, Consulta3.VibLA, Consulta3.VibLL, Consulta3.Temperatura
      FROM
      (
      SELECT Consulta1.TAG, Consulta2.Motor, Consulta2.kW, Consulta1.Expr1, Consulta2.VibLL, Consulta2.VibLA, Consulta2.Proceso, Consulta2.Temperatura, Consulta2.Toma
      FROM
      (
      SELECT Vibraciones_1.TAG, Max(Vibraciones_1.Fecha) AS Expr1
      FROM vibraciones AS Vibraciones_1
      GROUP BY Vibraciones_1.TAG

      )
      as Consulta1 INNER JOIN
      (
      SELECT motores.TAG, motores.kW, vibraciones.VibLL, vibraciones.VibLA, vibraciones.Fecha, vibraciones.Toma, motores.Proceso, vibraciones.Temperatura, motores.Motor
      FROM motores INNER JOIN vibraciones ON motores.TAG = vibraciones.TAG

      )
      as Consulta2 ON (Consulta1.Expr1 = Consulta2.Fecha) AND (Consulta1.TAG = Consulta2.TAG)
      )
      as Consulta3 WHERE (((Consulta3.kW)<15) AND ((Consulta3.VibLA)>2.8 And (Consulta3.VibLA)<4.5)) OR (((Consulta3.kW)<15) AND ((Consulta3.VibLA)>=4.5)) OR (((Consulta3.kW)<15) AND ((Consulta3.VibLL)>2.8 And (Consulta3.VibLL)<4.5)) OR (((Consulta3.kW)<15) AND ((Consulta3.VibLL)>=4.5)) OR (((Consulta3.kW)<75 And (Consulta3.kW)>=15) AND ((Consulta3.VibLA)>=2.8 And (Consulta3.VibLA)<7.1)) OR (((Consulta3.kW)<75 And (Consulta3.kW)>=15) AND ((Consulta3.VibLA)>=7.1)) OR (((Consulta3.kW)<75 And (Consulta3.kW)>=15) AND ((Consulta3.VibLL)>=2.8 And (Consulta3.VibLL)<7.1)) OR (((Consulta3.kW)<75 And (Consulta3.kW)>=15) AND ((Consulta3.VibLL)>=7.1)) OR (((Consulta3.kW)>=75) AND ((Consulta3.VibLA)>4.5 And (Consulta3.VibLA)<11)) OR (((Consulta3.kW)>=75) AND ((Consulta3.VibLA)>=11)) OR (((Consulta3.kW)>=75) AND ((Consulta3.VibLL)>4.5 And (Consulta3.VibLL)<11)) OR (((Consulta3.kW)>=75) AND ((Consulta3.VibLL)>=11))


      )

      as Riesgo WHERE Proceso ="'
      .$proceso.
      '")');
      // Bind the parameter
      $oCommand->bindParam(':Riesgo', $Riesgo, PDO::PARAM_STR);

      $oCDbDataReader = $oCommand->queryAll(); // Run query and get all results in a CDbDataReader
     */
    ?>
