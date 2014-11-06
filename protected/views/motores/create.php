<?php
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
        border-radius:2px;


    }



</style>

<script type="text/javascript">
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('proceso' => 'js:document.getElementById("Motores_Proceso").value'),
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
    'type' => 'GET', //request typem
    'data' => array('area' => 'js:document.getElementById("Motores_Area").value'),
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
    
    function updateGridMotores()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("Motores_Area").value', 'equipo' => 'js:document.getElementById("Motores_Equipo").value'),
    'update' => '#gridMotores',
    'url' => CController::createUrl('/motores/dynamicMotores'), //url to call.
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
        $('#Motores_Area').html(data.value1);
        updateFieldEquipo();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#Motores_Equipo').html(data.value1);
        updateGridMotores();
    };


</script>

<?php
$form = $this->beginWidget('CActiveForm', array(
    'id' => 'motores-form',
    'enableAjaxValidation' => true,
    'htmlOptions' => array('enctype' => 'multipart/form-data'),
        ));
?>

<?php
$this->breadcrumbs = array(
    'Motores' => array(Yii::t('app', 'index')),
    Yii::t('app', 'Crear'),
);

$this->menu = array(
    array('label' => 'Lista de Motores', 'url' => array('index')),
    array('label' => 'Gestionar Motores', 'url' => array('admin')),
);
?>

<?php $this->setPageTitle(' Crear Motor '); ?>

<div class="form">
    <div name="myDiv" id="myDiv" class="forms50cb">
        <div styler="padding-left:5px;padding-right:5px;padding-top:3px;width:400px!important;margin-left:0px;margin-bottom:5px !important;color:#961C1F">

            <div styler="padding-left:5px;width:400px">

                <div>
                    <div styler="width:50%;">
                        <b>Area:</b>
                        <?php
                        if (isset($modeloId)) {
                            $model->Proceso = $modeloId->Proceso;
                        }
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                        echo CHtml::activeDropDownList($model, 'Proceso', CHtml::listData(Estructura::model()->findAllbySql(
                                                'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
                                ), array(
                            //'onfocus' => 'updateFieldArea()',
                            'onchange' => 'updateFieldArea()',
                            'style' => 'width:100%;',
                            'class' => 'select',
                                )
                        );
                        ?>
                        <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
                    </div>
                    <div styler="width:50%;">
                        <b>Proceso:</b>
<?php
if (isset($modeloId)) {
    $model->Area = $modeloId->Area;
}
// dibuja el dropDownList de Area, dependiendo del proceso selecccionado
echo CHtml::activeDropDownList($model, 'Area', CHtml::listData(Estructura::model()->findAllbySql(
                        'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . (isset($modeloId) ? $model->Proceso : "ELABORACION") . '" ORDER BY Area ASC', array()), 'Area', 'Area'
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
                </div>
            </div>


            <div styler="padding-left:10px;width:400px">

                <div>
                    <div styler="width:100%;">
                        <b>Equipo:</b>
<?php
if (isset($modeloId)) {
    $model->Equipo = $modeloId->Equipo;
}
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
echo CHtml::activeDropDownList($model, 'Equipo', CHtml::listData(Estructura::model()->findAllbySql(
                        'SELECT Equipo FROM estructura WHERE Area="' . (isset($modeloId) ? $model->Area : "FILTRACION") . '" ORDER BY Equipo ASC', array()), 'Equipo', 'Equipo'
        ), array(
    //'onfocus' => 'updateFieldEquipo()',
    'onchange' => 'updateGridMotores()',
    'style' => 'width:100%;',
    'class' => 'select',
    'empty' => 'Seleccione el equipo',
        )
);
?>
                    </div>
                </div>
            </div>

            </fieldset>

        </div>    

<?php
echo $this->renderPartial('_form', array(
    'model' => $model,
    'modelArchivo' => $modelArchivo,
    'form' => $form
));
?>

        <div class="row buttons forms100c">
        <?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
        </div>

        <?php $this->endWidget(); ?>

    </div>
