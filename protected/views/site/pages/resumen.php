<style type="text/css">
  
  div.loading {
    background-color: #FFFFFF;
    background-image: url('/images/loading.gif');
    background-position:  100px;
    background-repeat: no-repeat;
    opacity: 1;
}
div.loading * {
    opacity: .8;
}

     
    .select{
        
        width: 30px;
        padding-left:2px;
        background:#ffffff;
        background-color:#ffffff;
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
    -webkit-border-radius: 3px;
    border-radius:3px;
     
    }   
    
    .back{

               
       padding-left:10px


    }

</style>


<?php
$this->layout='responsiveLayout';//antes era un column1
$this->pageTitle=Yii::app()->name . ' - Resumen de Resultados';
$this->breadcrumbs=array(
	'Resumen de resultados',
);
?>

<div class="forms50c">

 <b>Area:</b>
<?php
$valor=isset($_GET['area'])?$_GET['area']:"";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
echo CHtml::dropDownList(
        'proceso', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                        'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
        ), 
        array(
            'ajax' => array(
                'type' => 'GET', //request type
                'data' => array('proceso' => 'js:document.getElementById("proceso").value'),
                'url' => CController::createUrl('/site/dynamicResumen'), //url to call.
                'update' => '#divResumen', //selector to update
            ),
            'style' => 'width:100%;',
            'class'=>'select',
            'empty'=>'Seleccione el Area'
        )
);
?>
 

</div>    
<script type="text/javascript">
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateGrid()
    {
                
<?php
    echo CHtml::ajax(array (
                'type' => 'GET', //request type
                'data' => array('proceso' => 'js:document.getElementById("proceso").value'),
                'url' => CController::createUrl('/site/dynamicResumen'), //url to call.
                'update' => '#divResumen',
        //selector to update
        )
    );
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;

        return false;
    }
</script>

<?php



    // dibuja el gridview si hay un Id
    if (isset($_GET['area'])) 
    {
            echo "<script language=javascript>updateGrid()</script>";
        }
?>
<div id="divResumen"> 
    
    </div>
    

<!--
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

as Riesgo WHERE (Proceso = 'SERVICIOS')
--->














