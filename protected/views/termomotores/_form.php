<style type="text/css">

    .filtro2{
        width:250px;
        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;

    }

</style>




<div class="forms50cb">
    <div class="row forms50c">
        <?php $today = date("Y-m-d"); ?>

        <?php echo $form->labelEx($model, 'Fecha', array('styler' => 'text-align:left;')); ?> 
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
            // 'value' => $today,
            'themeUrl' => '/themes',
            'theme' => 'calendarioCbm',
            'htmlOptions' => array('style' => 'width:85%;'),
            'options' => array(
                'dateFormat' => 'yy-mm-dd',
                'showButtonPanel' => true,
                "yearRange" => '1995:2070',
                'changeYear' => true,
                'buttonImage' => '/images/calendar.png',
                'showOn' => "both",
                'buttonText' => "Seleccione la fecha",
                'buttonImageOnly' => true,
            ) // jquery plugin options
        ));
        ?>

    </div>
    <div class="row forms50c">
        <?php echo $form->labelEx($model, 'OT', array('styler' => 'text-align:left;')); ?>
        <?php echo $form->textField($model, 'OT', array()); ?> 
        <?php echo $form->error($model, 'OT'); ?>
    </div>


    <div class="row" >
        <?php
        echo $form->labelEx($model, 'Buscar Informe', array('styler' => 'text-align:left;'));
        echo $form->fileField($modelArchivo, 'nombre', array('style' => 'height:21px'));
        echo $form->error($modelArchivo, 'nombre');
        ?>

    </div>

    <div class="row forms50c">
        <?php echo $form->labelEx($model, 'Analista', array('styler' => 'text-align:left;')); ?>
        <?php echo $form->textField($model, 'Analista', array('size' => 50, 'maxlength' => 50, 'styler' => 'width:230px;')); ?> 
        <?php echo $form->error($model, 'Analista'); ?>
    </div> 


    <div class="row forms50c">
        <?php echo $form->labelEx($model, 'Tamano', array('styler' => 'text-align:left;')); ?>
        <?php echo $form->textField($model, 'Tamano', array('styler' => 'width:230px;')); ?> 
        <?php echo $form->error($model, 'Tamano'); ?>
    </div>
    <div class="row forms50c">
        <?php echo $form->labelEx($model, 'Estado', array('styler' => 'text-align:left;')); ?>
        <?php
        echo $form->dropDownList($model, 'Estado', array(
            0 => 'Adecuado',
            1 => 'Atención Requerida',
            2 => 'Malo',
                ), array('styler' => 'width:230px;'));
        //textField($model, 'Estado', array('size' => 50, 'maxlength' => 50)); 
        ?>
        <?php echo $form->error($model, 'Estado'); ?>
    </div>
    <div class="row forms50c">
        <?php echo $form->labelEx($model, 'Observaciones', array('styler' => 'text-align:left;')); ?>
        <?php echo $form->textArea($model, 'Observaciones', array('styler' => 'width:230px;')); ?>
        <?php echo $form->error($model, 'Observaciones'); ?>
    </div>                    
</div>

<div class="forms50cb">
    <?php echo $form->errorSummary($model); ?>
    <img src="/images/TermoCriterios.png" style="width:90%;max-width: 400px;"/>
</div>
























