<div class="wide form">

    <?php
    $form = $this->beginWidget('CActiveForm', array(
        'action' => Yii::app()->createUrl($this->route),
        'method' => 'get',
            ));
    ?>


    <div class="row">
<?php echo $form->label($model, 'Analista'); ?>
<?php echo $form->textField($model, 'Analista', array('size' => 50, 'maxlength' => 50)); ?>
    </div>

    <div class="row">
<?php echo $form->label($model, 'OT'); ?>
<?php echo $form->textField($model, 'OT'); ?>
    </div>


    <div class="row">
<?php echo $form->label($model, 'ZI'); ?>
<?php echo $form->textField($model, 'ZI'); ?>
    </div>

    <div class="row">
        <?php $today = date("Y-m-d"); ?>

        <?php echo $form->label($model, 'Fecha'); ?>
        <?php
//TODO: Corregir fecha al actualizar
        if (defined($model->Fecha))
            $today = $model->Fecha;

        Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
        $this->widget('CJuiDateTimePicker', array(
            'model' => $model, //Model object
            'attribute' => 'Fecha', //attribute name
            'mode' => 'datetime', //use "time","date" or "datetime" (default)
            'language' => 'es',
            'value' => $today,
            'themeUrl' => '/themes',
            'theme' => 'calendarioCbm',
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
        ?>
    </div>


    <div class="row">
        <?php echo $form->label($model, 'Equipo'); ?>
        <?php echo $form->textField($model, 'Equipo', array('size' => 50, 'maxlength' => 50)); ?>
    </div>

    <div class="row buttons">
    <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
            </div>
   
    <?php $this->endWidget(); ?>



    <!--
    
     <div class="row">
    <?php echo $form->label($model, 'Proceso'); ?>
    <?php echo $form->textField($model, 'Proceso', array('size' => 50, 'maxlength' => 50)); ?>
            </div>
    
            <div class="row">
    <?php echo $form->label($model, 'Area'); ?>
    <?php echo $form->textField($model, 'Area', array('size' => 50, 'maxlength' => 50)); ?>
            </div>
         <div class="row">
    <?php echo $form->label($model, 'Descripcion'); ?>
    <?php echo $form->textArea($model, 'Descripcion', array('rows' => 6, 'cols' => 50)); ?>
            </div>
            <div class="row">
    <?php echo $form->label($model, 'Reporte'); ?>
    <?php echo $form->textField($model, 'Reporte'); ?>
            </div>
    
            <div class="row">
    <?php echo $form->label($model, 'Path'); ?>
    <?php echo $form->textField($model, 'Path', array('size' => 60, 'maxlength' => 255)); ?>
            </div>
    
            <div class="row">
    <?php echo $form->label($model, 'Presion'); ?>
    <?php echo $form->textField($model, 'Presion'); ?>
            </div>
    
            <div class="row">
    <?php echo $form->label($model, 'Decibeles'); ?>
    <?php echo $form->textField($model, 'Decibeles'); ?>
            </div>
        
           
    
            <div class="row">
    <?php echo $form->label($model, 'Gas'); ?>
    <?php echo $form->textField($model, 'Gas', array('size' => 50, 'maxlength' => 50)); ?>
            </div>
    
            <div class="row">
    <?php echo $form->label($model, 'Tamano'); ?>
    <?php echo $form->textField($model, 'Tamano'); ?>
            </div>
    
            <div class="row">
    <?php echo $form->label($model, 'CFM'); ?>
    <?php echo $form->textField($model, 'CFM'); ?>
            </div>
    
            <div class="row">
    <?php echo $form->label($model, 'COSTO'); ?>
    <?php echo $form->textField($model, 'COSTO'); ?>
            </div>
    
            <div class="row">
    <?php echo $form->label($model, 'Corregido'); ?>
    <?php echo $form->checkBox($model, 'Corregido'); ?>
            </div>
    --->
        
</div><!-- search-form -->
