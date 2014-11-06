<?php
//para leer el param get y con el reconfigurar los dropdown
$sufix = "";
if (isset($_GET['proceso'])) {
    $modeloId = Estructura::model()->findByAttributes(array("Area" => $_GET['proceso']));
    $sufix = "?id=" . $modeloId->Area;
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


        padding-left:605px


    }




</style>



</style> 

<script type="text/javascript">
    
    function updateGridEstructura()
    {
        
        $('#linkNuevo').html('<a href="/index.php/estructura/create?id='+document.getElementById("area").value+'">Nuevo Equipo</a>'); 
      
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/estructura/dynamicEstructura'), //url to call.
    'update' => '#divEstructura', //selector to update
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
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
        //  $("#area").trigger("liszt:updated");
        // $("#equipo").trigger("liszt:updated");
        updateGridEstructura();
        //updateFieldEquipo();
        //updateGridTableros();
        
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#equipo').html(data.value1);
        //updateGridReportes();
        //  $("#equipo").trigger("liszt:updated");
        // $("#motor").trigger("liszt:updated");
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdownVacio(data)
    {   
        updateFieldEquipoComment();
        updateGridReportesArea();
        //  $("#equipo").trigger("liszt:updated");
    };
    
    // función que coloca el dropdown de Equipo  con comentario únicamente
    function updateAreaDropdownComment(data)
    {
        $('#equipo').html(data.value1);
        //   $("#motor").trigger("liszt:updated");
        //updateFieldEquipo();
    };
    


</script>


<?php
$this->breadcrumbs = array(
    'Equipos ' => array(Yii::t('app', 'index')),
    Yii::t('app', ''),
);

$this->menu = array(
    array('label' => Yii::t('app', 'Lista de Equipos'), 'url' => array('index')),
    
);
//TODO: provisional: para uso de roles de admin, ingeniero y usuario.
$esAdmin = 0;
$esIngeniero = 0;
if (!Yii::app()->user->isGuest) {
    $modeloU = Usuarios::model()->findBySql('select * from usuarios where Username="' . Yii::app()->user->name . '"');
}
if (isset($modeloU)) {
    $esAdmin = $modeloU->Es_administrador;
    $esIngeniero = $modeloU->Es_analista;
    if ($esAdmin)
        $esIngeniero = 1;
}
if ($esIngeniero)
    array_push($this->menu, array('label' => Yii::t('app', 'Nuevo Equipo'),
        'url' => array('create' . $sufix),
        'itemOptions' => array('id' => 'linkNuevo')));


Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('estructura-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
?>

<?php $this->setPageTitle (' Gestionar&nbsp;Equipos'); ?>

 <div name="myDiv" id="myDiv" class="forms100cb" class="forms50cb"> 

     <table style="width:100%;">
   
  
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
                'onchange' => 'updateFieldArea();',
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
                                            'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC' : 'SELECT DISTINCT Area FROM estructura WHERE Proceso="ELABORACION" ORDER BY Area ASC', array()), 'Area', 'Area'
                    ), array(
                'onchange' => 'updateGridEstructura();',
                'style' => 'width:100%;',
                'class' => 'select',
                'empty' => 'Seleccione el proceso',
                    )
            );
            ?>
        </td>
    </tr>
</table>


<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'), '#', array('class' => 'search-button')); ?>
<div class="search-form" style="display:none">
    <?php
    $this->renderPartial('_search', array(
        'model' => $model,
    ));
    ?>
</div>
</div>
<div id="divEstructura" name="divEstructura">



    <?php
// dibuja el gridview si hay un Id
    if (isset($modeloId)) {
        echo "<script language=javascript>updateGridEstructura()</script>";
    } else {

        /*
          $this->widget('zii.widgets.grid.CGridView', array(
          'id' => 'clinicas-grid',
          'dataProvider' => $model->search(),
          // 'filter' => $model,
          'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
          'columns' => array(
          'Codigo',
          'Equipo',
          array(
          'class' => 'CButtonColumn',
          ),
          ),
          ));
         */
    }
    ?>


</div>
