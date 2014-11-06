<!--
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
    'update' => '#gridFechas',
    'url' => CController::createUrl('/motores/dynamicFechas'), //url to call.
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
--->


<div class="forms100cb"> 
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'OT'); ?>
        <?php echo $form->textField($model, 'OT'); ?>
        <?php echo $form->error($model, 'OT'); ?>
    </div>
    <div class="forms33c">
        <?php $today = date("Y-m-d"); ?>

        <?php echo $form->labelEx($model, 'Fecha'); ?>
        <?php
//Fecha inicial
        $today = date("Y-m-d H:i:s");
        if (isset($model->Fecha))
            $today = $model->Fecha;
        else
            $model->Fecha = $today;
//fin Fecha inicial
        if (defined($model->Fecha))
            $today = $model->Fecha;

        Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
        $this->widget('CJuiDateTimePicker', array(
            'model' => $model, //Model object
            'attribute' => 'Fecha', //attribute name
            'mode' => 'datetime', //use "time","date" or "datetime" (default)
            'language' => 'es',
            //  'value' => $today,
            'themeUrl' => '/themes',
            'theme' => 'calendarioCbm',
            'htmlOptions' => array('style' => 'width:80%;'),
            'options' => array(
                'dateFormat' => 'yy-mm-dd',
                'showButtonPanel' => true,
                "yearRange" => '1995:2070',
                'changeYear' => true,
                'buttonImage' => '/images/calendar.png',
                'showOn' => "both",
                'buttonText' => "Seleccione la fecha",
                'buttonImageOnly' => true
            ) // jquery plugin options
        ));
        ;
        ?>
    </div>


    <div class="forms33c">
        <?php echo $form->errorSummary($model); ?>
        <?php echo $form->labelEx($model, 'VibLL'); ?>
        <?php echo $form->textField($model, 'VibLL', array('size' => 19, 'maxlength' => 19)); ?>
        <?php echo $form->error($model, 'VibLL'); ?>
    </div>

    <div class="forms33c">
        <?php echo $form->labelEx($model, 'VibLA'); ?>
        <?php echo $form->textField($model, 'VibLA', array('size' => 18, 'maxlength' => 18)); ?>
        <?php echo $form->error($model, 'VibLA'); ?>
    </div>

    <div class="forms33c">
        <?php echo $form->labelEx($model, 'Temperatura'); ?>
        <?php echo $form->textField($model, 'Temperatura', array('size' => 18, 'maxlength' => 18)); ?>
        <?php echo $form->error($model, 'Temperatura'); ?>
    </div> 
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'Estado'); ?>
        <?php
        echo $form->dropDownList($model, 'Estado', array(
            0 => 'Adecuado',
            1 => 'Atención Requerida',
            2 => 'Malo',
                ), array('styler' => ''));
//textField($model, 'Estado', array('size' => 50, 'maxlength' => 50)); 
        ?>
        <?php echo $form->error($model, 'Estado'); ?>
    </div>


    <div class="forms100c">
        <?php echo $form->labelEx($model, 'Observaciones'); ?>
        <?php echo $form->textArea($model, 'Observaciones', array('styler' => 'width:100%;')); ?>
        <?php echo $form->error($model, 'Observaciones'); ?>
    </div>                    

</div>


