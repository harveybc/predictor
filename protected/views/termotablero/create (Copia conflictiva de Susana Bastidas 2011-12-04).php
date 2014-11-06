<?php
    //para leer el param get y con el reconfigurar los dropdown
    if (isset($_GET['id']))
    {
        $modeloId = Tableros::model()->findByAttributes(array ("TAG" => $_GET['id']));
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
     
    .select{
                       
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
    'url' => CController::createUrl('/tableros/dynamicTableroDropDown'), //url to call.
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
       
    function updateFieldTablero()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/termotablero/dynamicTableros'), //url to call.
    //'update' => '#divTableros', //selector to update
    'success' => 'updateTableros',
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
       function updateTableros(data)
    {
        $('#Termotablero_TAG').html(data.value1);
       
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
        $('#Termotablero_TAG').html(data.value1);
        //updateGridReportes();
    };
</script>


<?php
$this->breadcrumbs=array(
	'Informes de Tableros Eléctricos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Informes', 'url'=>array('index')),
	array('label'=>'Gestionar Informes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Informe de Tablero Eléctrico'); ?>



<div class="form">
<div name="myDiv" id="myDiv" class="forms50cb">
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
                                                'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC' : 'SELECT DISTINCT Area FROM estructura WHERE Proceso="ELABORACION" ORDER BY Area ASC', array()), 'Area', 'Area'
                        ), array(
                    'onchange' => 'updateFieldTablero()',
                    'style' => 'width:100%;',
                    'class' => 'select',
                    'empty' => 'Seleccione un proceso',
                        )
                );
                ?>
            </td>
        </tr>
    </table>
    
    
<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'termotablero-form',
	'enableAjaxValidation'=>true,
)); 
?>
    <table style="width:100%; ">
        <tr>
            <td>
                <b>Tablero:<br/></b>
                <div id="divTableros" name="divTableros">
                    <?php
                        if (isset($modeloId))
                            $model->TAG=$modeloId->TAG;
                        // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                        echo CHtml::activeDropDownList($model,
                            'TAG', CHtml::listData(Tableros::model()->findAllbySql(
                                    'SELECT TAG,concat(TAG," - ",Tablero) as Tablero FROM tableros WHERE Area="' .(isset($modeloId)? $modeloId->Area:"FILTRACION" ). '" ORDER BY Tablero ASC',
                                                                                                    array ()),
                                                                                                    'TAG',
                                                                                                    'Tablero'
                            ),
                                       array(
                                            //'onfocus' => 'updateGrid()',
                                            //'onchange' => 'updateGrid()',
                                            'style' => 'width:100%;',
                                            'class'=>'select',
                                                )
                                        );
                    ?>
                </div>
            </td>
        </tr>
    </table>

</div>
   
<?php
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
