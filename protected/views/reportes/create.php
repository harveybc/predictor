<?php
//para leer el param get y con el reconfigurar los dropdown
if (isset($_GET['id'])) {
    $modeloId = Estructura::model()->findByAttributes(array("Equipo" => $_GET['id']));
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


    .selecta{

        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:3px;
    }
</style> 


<script type="text/javascript">
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
        
        
      
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('proceso' => 'js:document.getElementById("Reportes_Proceso").value'),
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
    'data' => array('area' => 'js:document.getElementById("Reportes_Area").value'),
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
    'data' => array('area' => 'js:document.getElementById("Reportes_Area").value'),
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
    'data' => array('area' => 'js:document.getElementById("Reportes_Area").value'),
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
    'data' => array('area' => 'js:document.getElementById("Reportes_Area").value', 'equipo' => 'js:document.getElementById("Reportes_Equipo").value'),
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
    'data' => array('area' => 'js:document.getElementById("Reportes_Area").value'),
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
        $('#Reportes_Area').html(data.value1);
        updateFieldEquipo();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#Reportes_Equipo').html(data.value1);
        //updateGridReportes();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdownVacio(data)
    {   
        updateFieldEquipoComment();
        updateGridReportesArea();
    };
    
    // función que coloca el dropdown de Equipo  con comentario únicamente
    function updateAreaDropdownComment(data)
    {
        $('#Reportes_Equipo').html(data.value1);
        //updateFieldEquipo();
    };
    


</script>

<?php
$this->breadcrumbs = array(
    'Reportes' => array(Yii::t('app', 'index')),
    Yii::t('app', 'Crear'),
);

$this->menu = array(
    array('label' => 'Lista de Reportes', 'url' => array('index')),
    array('label' => 'Gestionar Reportes', 'url' => array('admin')),
);
?>

<?php $this->setPageTitle(' Crear reporte de fuga de gases (Ultrasonido)'); ?>
<div class="form">

<?php
$form = $this->beginWidget('CActiveForm', array(
    'id' => 'reportes-form',
    'enableAjaxValidation' => true,
    'htmlOptions' => array('enctype' => 'multipart/form-data'),
        ));
?>
    <div name="myDiv" id="myDiv" class="forms100cb">
        <div class="forms50c">
            <b>Area:</b>
    <?php
    if (isset($modeloId))
        $model->Proceso = $modeloId->Proceso;
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
    echo CHtml::activeDropDownList($model, 'Proceso', CHtml::listData(Estructura::model()->findAllbySql(
                            'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
            ), array(
        //'onfocus' => 'updateFieldArea()',
        'onchange' => 'updateFieldArea()',
        'style' => 'width:100%;',
        'class' => 'select',
        'empty' => 'Seleccione el área',
            )
    );
    ?>
            <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
        </div>

        <div class="forms50c">
            <b>Proceso:</b>
            <?php
            if (isset($modeloId))
                $model->Area = $modeloId->Area;
            // dibuja el dropDownList de Area, dependiendo del proceso selecccionado
            echo CHtml::activeDropDownList($model, 'Area', CHtml::listData(Estructura::model()->findAllbySql(
                                    'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . (isset($modeloId) ? $modeloId->Proceso : "ELABORACION") . '" ORDER BY Area ASC', array()), 'Area', 'Area'
                    ), array(
                //'onfocus' => 'updateFieldArea()',
                'onchange' => 'updateFieldEquipo()',
                'style' => 'width:100%;',
                'class' => 'select',
                'empty' => 'Seleccione el proceso',
                    )
            );
            ?>
        </div>
        <div class="row">
            <b>Equipo:</b>
            <?php
            if (isset($modeloId)) {
                $model->Equipo = $modeloId->Equipo;
            }
            // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso

            echo CHtml::activeDropDownList($model, 'Equipo', CHtml::listData(Estructura::model()->findAllbySql(
                                    'SELECT Equipo FROM estructura WHERE Area="' . (isset($modeloId) ? $modeloId->Area : "FILTRACION" ) . '" ORDER BY Equipo ASC', array()), 'Equipo', 'Equipo'
                    ), array(
                //'onfocus'  => 'updateFieldEquipo()',
                //'onchange' => 'updateGridReportes()',
                'style' => 'width:100%;',
                'class' => 'select',
                'empty' => 'Seleccione el equipo',
                    )
            );
            ?>
        </div>
    </div>


            <?php
            echo $this->renderPartial('_form', array(
                'model' => $model,
                'modelArchivo' => $modelArchivo,
                'form' => $form
            ));
            ?>

    <div class="row buttons forms100c">
            <?php echo CHtml::submitButton(Yii::t('app', 'Crear'));
            ?>
    </div>

            <?php $this->endWidget(); ?>

</div>
