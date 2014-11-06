<?php
    if  (isset($_GET['id']))
    {
       $modeloId=Motores::model()->findByAttributes(array("TAG"=>$_GET['id']));
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

    function updateFieldMotor()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('equipo' => 'js:document.getElementById("equipo").value'),
    'url' => CController::createUrl('/motores/dynamicFMotor'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateMotorDropdown',
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

    function updateGridFechas()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('TAG' => 'js:document.getElementById("motor").value'),
    'update'=>'#gridFechas',
    'url' => CController::createUrl('/vibraciones/dynamicFechas'), //url to call.
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
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#equipo').html(data.value1);
        updateFieldMotor();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateMotorDropdown(data)
    {
        $('#Vibraciones_TAG').html(data.value1);
        //updateGridFechas();
    };


</script>


<?php
$this->breadcrumbs=array(
	'Vibraciones'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Registros', 'url'=>array('index')),
	array('label'=>'Gestionar Registros', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Registro de Vibraciones y Temperatura '); ?>
<div name="myDiv" id="myDiv" class="forms100cb">
<div class="form">
            <div class="forms50c">
                <b>Area:</b>
<?php
    $valor=isset($modeloId)?$modeloId->Proceso:"";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
echo CHtml::dropDownList(
        'proceso', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                        'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
        ), array(
    //'onfocus' => 'updateFieldArea()',
    'onchange' => 'updateFieldArea()',
    'style' => 'width:100%;',
    'class'=>'select',
    
        )
);
?>
                <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
            </div>
            <div class="forms50c">
                <b>Proceso:</b>
                <?php
                    $valor=isset($modeloId)?$modeloId->Area:"";
                // dibuja el dropDownList de Area, dependiendo del proceso selecccionado
                echo CHtml::dropDownList(
                        'area', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                        'SELECT DISTINCT Area FROM estructura WHERE Proceso="'.(isset($modeloId)?$modeloId->Proceso:"ELABORACION").'" ORDER BY Area ASC', array()), 'Area', 'Area'
                        ), array(
                    //'onfocus' => 'updateFieldArea()',
                    'onchange' => 'updateFieldEquipo()',
                    'style' => 'width:100%;',
                    'class'=>'select',
                    'empty'=>'Seleccione el proceso',
                     
                        )
                );
                ?>
            </div>
            <div>
                <b>Equipo:</b>
                <?php
                 $valor=isset($modeloId)?$modeloId->Equipo:"";
                // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                echo CHtml::dropDownList(
                        'equipo', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                        'SELECT Equipo FROM estructura WHERE Area="'.(isset($modeloId)?$modeloId->Area:"FILTRACION").'" ORDER BY Equipo ASC', array()), 'Equipo', 'Equipo'
                        ), array(
                    //'onfocus' => 'updateFieldEquipo()',
                    'onchange' => 'updateFieldMotor()',
                    'style' => 'width:100%;',
                    'class'=>'select',
                    'empty'=>'Seleccione el equipo',
                        )
                );
                ?>
            </div>
            <div>
                <b>Motor:</b>
                <?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'vibraciones-form',
	'enableAjaxValidation'=>true,
)); 
 ?>
                <?php
                $valor=isset($modeloId)?$modeloId->TAG:"";
                // dibuja el dropDownList de Motores, 
                echo CHtml::dropDownList(
                        'Vibraciones[TAG]', $valor, CHtml::listData(Motores::model()->findAllbySql(
                                        'SELECT TAG, CONCAT(TAG," - ",Motor) as Motor FROM motores WHERE Equipo="'.(isset($modeloId)?$modeloId->Equipo:"ANILLO DE CONTRAPRESION").'" ORDER BY TAG ASC', array()), 'TAG', 'Motor'
                        ), array(
                    //'onfocus' => 'updateFieldEquipo()',
                    //'onchange' => 'updateGridFechas()',
                    'style' => 'width:100%;',
                    'class'=>'select',
                    'empty'=>'Seleccione el motor',
                        )
                );
                ?>
            </div>
        <div>
    </div>
    
    </div>
    </div>
  
<?php
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	));
?>
<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>


