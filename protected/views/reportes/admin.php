<?php
//para leer el param get y con el reconfigurar los dropdown
$sufix = "";
if (isset($_GET['id'])) {
    $modeloId = Estructura::model()->findByAttributes(array("id" => $_GET['id']));
    if (isset($modeloId))
    $sufix = "?id=" . $modeloId->Equipo;
}
?>
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

    .select_admin{

        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;
        


    }
    .back{


        padding:0px;
        margin:0px;

    }




</style> 


<script type="text/javascript">
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
        
        
      
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('proceso' => 'js:document.getElementById("proceso").value'),
    'url' => CController::createUrl('/motores/dynamicArea'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateAreaDropdown',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
    
    function updateFieldEquipo()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/motores/dynamicEquipo'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateEquipoDropdown',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
   
    function updateFieldEquipoPre(equipo)
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/motores/dynamicEquipo'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateEquipoDropdown',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //setTimeout($('#equipo').attr('value',equipo),200);
        return false;
    }
   
    function updateFieldEquipoVacio()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/motores/dynamicEquipoVacio'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateEquipoDropdownVacio',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
    
    function updateFieldEquipoComment()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/motores/dynamicEquipoVacio'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateAreaDropdownComment',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    function updateGridReportes()
    {
        $('#linkNuevo').html('<a href="/index.php/reportes/create?id='+document.getElementById("equipo").value+'">Nuevo Reporte</a>');
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value', 'equipo' => 'js:document.getElementById("equipo").value'),
    'update' => '#divGridReportes',
    'url' => CController::createUrl('/reportes/dynamicReportes'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    function updateGridReportesArea()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'update' => '#divGridReportes',
    'url' => CController::createUrl('/reportes/dynamicReportesArea'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }


    // función que cambia los datos de el Dropdownlist de Area
    function updateAreaDropdown(data)
    {
        $('#area').html(data.value1);
        updateFieldEquipo();
        // actualiza el echosen de area
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#equipo').html(data.value1);
        updateGridReportesArea();
        
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdownVacio(data)
    {   
        //updateFieldEquipoComment();
        //updateGridReportesArea();
    };
    
    // función que coloca el dropdown de Equipo  con comentario únicamente
    function updateAreaDropdownComment(data)
    {
        // $('#equipo').html(data.value1);
        //updateFieldEquipo();
        
    };
    function updateLinkCreate()
    {
        $('#equipo').html(data.value1);
    }


</script>



<?php
$this->breadcrumbs = array(
    'Reportes' => array(Yii::t('app', 'index')),
    Yii::t('app', 'Gestionar'),
);

$this->menu = array(
    array('label' => Yii::t('app', 'Instrucciones'), 'url' => array('/Archivos/displayArchivo?id=24')),
    array('label' => Yii::t('app', 'Lista de Reportes'), 'url' => array('index')),
    array('label' => Yii::t('app', 'Nuevo Reporte'),
        'url' => array('create' . $sufix),
        'itemOptions' => array('id' => 'linkNuevo')),
);

Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('reportes-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
?>


<?php $this->setPageTitle(' Gestionar&nbsp;reportes de fugas de gases (Ultrasonido)'); ?>

<div name="myDiv" id="myDiv" class="forms100cb">
<form>
    <table style="width:100%; ">
        <tr>
            <td style="width:50%;">
                <b>Area:</b>
                <?php
                $valor = isset($modeloId) ? $modeloId->Proceso : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                echo CHtml::dropDownList(
                        'proceso', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                        'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
                        ), array(
                    //'onfocus' => 'updateFieldArea()',
                    'onchange' => 'updateFieldArea()',
                    'style' => 'width:100%;',
                    'class' => 'select'
                        )
                );
                ?>
                <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
            </td>
            <td style="width:50%;">
                <b>Proceso:</b>
                <?php
                $valor = isset($modeloId) ? $modeloId->Area : "";
// dibuja el dropDownList de Area, dependiendo del proceso selecccionado
                echo CHtml::dropDownList(
                        'area', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                        isset($modeloId) ?
                                                'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC' : 'SELECT DISTINCT Area FROM estructura WHERE Proceso="ELABORACION" ORDER BY Area ASC', array()
                                ), 'Area', 'Area'
                        ), array(
                    //'onfocus' => 'updateFieldArea()',

                    'onchange' => 'updateFieldEquipo()',
                    'style' => 'width:100%;',
                    'class' => 'select',
                    'empty' => 'Seleccione el proceso.'
                        )
                );
                ?>
            </td>
        </tr>
    </table>

    <table style="width:100%; ">
        <tr>
            <td>
                <b>Equipo:<br/></b>
                <?php
                $valor = isset($modeloId) ? $modeloId->Equipo : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                echo CHtml::dropDownList(
                        'equipo', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                        isset($modeloId) ?
                                                'SELECT Equipo FROM estructura WHERE Area="' . $modeloId->Area . '" ORDER BY Equipo ASC' : 'SELECT Equipo FROM estructura WHERE Area="FILTRACION" ORDER BY Equipo ASC', array()), 'Equipo', 'Equipo'
                        ), array(
                    //'onfocus' => 'updateFieldEquipo()',
                    'onchange' => 'updateGridReportes()',
                    'style' => 'width:100%;',
                    'class' => 'select',
                    'empty' => 'Seleccione el equipo para filtrar el resultado',
                        )
                );
                ?>
            </td>
        </tr>
    </table>
</form>
</div>





<div id="divGridReportes">
    Por favor, seleccione un equipo o un proceso.


<?php
if (isset($modeloId)) {
    echo "<script language=javascript>updateGridReportes()</script>";
} else {
    
    /*
    $this->widget('zii.widgets.grid.CGridView', array(
        'id' => 'clinicas-grid',
        'dataProvider' => $model->search(),
        // 'filter' => $model,
        'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
        'columns' => array(
        'Path',
        'Presion',
        'Decibeles',
        'Descripcion',
        'COSTO',
        array(
            'class' => 'CButtonColumn',
        ),
        ),
    ));

*/
}
?>


<br/>

<?php
/*
  // ejemplo de impresión de un resultado de consulta de SQL
  // compone la cadena de consulta
  $consultaSQL='
  SELECT COUNT(*) AS Expr1 FROM motores WHERE (Proceso=?)
  ';
  // se prepara el comando de SQL como en http://www.yiiframework.com/doc/guide/1.1/en/database.dao
  $command = Yii::app()->db->createCommand($consultaSQL);
  // se ejecuta la consulta y los resultados quedan en un arreglo de resultados $resultados[0] es el primero
  $resultados=$command->queryAll();
  // imprime todo el arreglo de resultados
  print_r($resultados);
  // ejemplo de uso de un campo de uno de los resultados (Equipo del resultado 0)
  echo "<br/>";
  echo '<b>Expr1='.$resultados[0]['Expr1'].'</b>'
 * 
 * 
 */
?>

</div>
















