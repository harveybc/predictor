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
    'url' => CController::createUrl('/aislamiento_tierra/dynamicFechas'), //url to call.
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
        $('#motor').html(data.value1);
        updateGridFechas();
    };


</script>


<?php
$this->breadcrumbs=array(
	'Vibraciones'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Registros'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Nuevo Registro'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('vibraciones-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp; registro de Vibraciones y Temperatura'); ?>


<form>

    <table style="padding-left:100px;padding-right:100px ;width:650px!important;margin-left:0px;margin-bottom:5px !important;">
        <tr> 
            <td>
                <b> Area:</b> 
                <select name="menu" style="width:200px">
                    <option value="0" selected>Servicios</option>
                    <option value="1">Elaboración</option>
                    <option value="2">Envase</option>
                </select>

            </td>

            <td>
                <b>Proceso</b >
                <select name="menu" style="width:200px;"> 
                    <option value="0" selected>Aire comprimido</option>
                    <option value="1">Generación frio</option>
                    <option value="2">Generación vapor</option>
                    <option value="3" selected>Planta CO2</option>
                    <option value="4">PTAP</option>
                    <option value="5">PTAR</option>
                </select>

            </td>

        </tr>

    </table>


    <table style="padding-left:50px;padding-right:90px;width:700px!important;margin-left:0px;margin-bottom:5px !important;">

        <tr>
            <td>
                <b>Equipo</b> 

                <select name="menu" style="width:550px"> >
                    <option value="0" selected>Aire comprimido</option>
                    <option value="1">Generación frio</option>
                    <option value="2">Generación vapor</option>
                    <option value="3" selected>Planta CO2</option>
                    <option value="4">PTAP</option>
                    <option value="5">PTAR</option>
                </select>

            </td>
        </tr>
        <tr>

            <td>
                <b>Motor Eléctrico</b> 

                <select name="menu" style="width:550px"> >
                    <option value="0" selected>Aire comprimido</option>
                    <option value="1">Generación frio</option>
                    <option value="2">Generación vapor</option>
                    <option value="3" selected>Planta CO2</option>
                    <option value="4">PTAP</option>
                    <option value="5">PTAR</option>
                </select>

            </td>
        </tr>
    </table> 
    
    </form>

 <div>

        </div>
    

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<fieldset style="padding-left:27px;width:300px!important;margin-left:0px;margin-bottom:5px !important; ">
    
    <table>
                        <tr>
                            <td>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'vibraciones-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
         'cssFile'=>'/themes/gridview/styles.css',
         'columns'=>array(
	//	'id',
	//	'Toma',
	//	'TAG',
		'Fecha',
	//	'OT',
	//	'VibLL',
		/*
		'VibLA',
		'Temperatura',
		*/
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>

                                 </td>

                            

                            </tr>


                    </table>
                    
                    </fieldset>

