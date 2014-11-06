

<?php echo $form->errorSummary($model); ?>
<?php
// if (isset($model->Toma)) 
//    echo '<input id="Aislamiento_acometida_Toma" name="Aislamiento_acometida[Toma]"'.$model->Toma.' type="hidden"/>';
?>
<?php echo $form->errorSummary($model); ?>
<?php
//if (isset($model->Toma)) 
//    echo '<input id="Aislamiento_fases_Toma" name="Aislamiento_fases[Toma]"'.$model->Toma.' type="hidden"/>';
if (isset($_GET['Toma'])) {
    $modeloAT = Aislamiento_tierra::model()->findByAttributes(array("Toma" => $_GET['Toma']));

    echo '<input id="Aislamiento_acometida_Toma" name="Aislamiento_acometida[Toma]" type="hidden" value="' . $modeloAT->Toma . '"/>';
    //echo '<input id="Aislamiento_acometida_Fecha" name="Aislamiento_acometida[Fecha]" type="hidden" value="'.$modeloAT->Fecha.'" />';
    $model->Fecha = $modeloAT->Fecha;
    echo '<input id="Aislamiento_acometida_TAG" name="Aislamiento_acometida[TAG]" type="hidden" value="' . $modeloAT->TAG . '" />';
}
?>

<div class="forms100cb" styler="padding-left:15px;padding-top:5px;width:440px!important;margin-left:0px;margin-bottom:5px !important;margin-top:12px !important;color:#961C1F;height:60px !important">
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'Orden de Trabajo'); ?>
        <?php echo $form->textField($model, 'OT', array('styler' => 'width:140px;')); ?> 
        <?php echo $form->error($model, 'OT'); ?>
    </div>
    <div class="forms33c" styler="width:200px;" >
        <?php $today = date("Y-m-d H:i:s"); ?>

        <?php echo $form->labelEx($model, 'Fecha'); ?>
        <?php
//TODO: Corregir fecha al actualizar
        if (isset($model->Fecha))
            $today = $model->Fecha;

        /*      $this->widget('zii.widgets.jui.CJuiDatePicker', array(
          'model' => '$model',
          'name' => 'Aislamiento_doomie',
          //  'id' => 'Aislamiento_tierra_Fecha',
          //  'dateFormat' => 'yy-mm-dd', // save to db format

          'language' => 'es',
          'value' => $today,
          //'value' => $today,
          'htmlOptions' => array('size' => 10, 'styler' => 'width:140px !important;margin_left:200px'),
          'options' => array(
          'dateFormat' => 'yy-mm-dd',
          'showButtonPanel' => true,
          "yearRange" => '1995:2070',
          'changeYear' => true,
          'altField' => '#Aislamiento_tierra_Fecha',
          'altFormat' => 'yy-mm-dd',
          'buttonImage'=>'/images/calendar.png',
          'showOn'=> "both",
          'buttonText'=>"Seleccione la fecha",
          'buttonImageOnly'=> true  //// show to user format
          ),
          'themeUrl' => '/themes',
          'theme' => 'calendarioCbm',
          )
          );
         */
        Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
        $this->widget('CJuiDateTimePicker', array(
            'model' => $model, //Model object
            'attribute' => 'Fecha', //attribute name
            'mode' => 'datetime', //use "time","date" or "datetime" (default)
            'language' => 'es',
            'value' => $today,
            'themeUrl' => '/themes',
            'theme' => 'calendarioCbm',
            'htmlOptions' => array('style' => 'width:80%;text-align:left;'),
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

</div>
<style>
    .forms33c{
        width:20%;
    }
</style>
<div class="forms100cb">
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'A025'); ?>
<?php echo $form->textField($model, 'A025'); ?>
        <?php echo $form->error($model, 'A025'); ?>
    </div>
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'A050'); ?>
<?php echo $form->textField($model, 'A050'); ?>
        <?php echo $form->error($model, 'A050'); ?>
    </div>
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'A1'); ?>
<?php echo $form->textField($model, 'A1'); ?>
        <?php echo $form->error($model, 'A1'); ?>
    </div>       
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'A2'); ?>
<?php echo $form->textField($model, 'A2'); ?>
        <?php echo $form->error($model, 'A2'); ?>
    </div>       
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'B025'); ?>
<?php echo $form->textField($model, 'B025'); ?>
        <?php echo $form->error($model, 'B025'); ?>
    </div>
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'B050'); ?>
<?php echo $form->textField($model, 'B050'); ?>
        <?php echo $form->error($model, 'B050'); ?>
    </div>
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'B1'); ?>
<?php echo $form->textField($model, 'B1'); ?>
        <?php echo $form->error($model, 'B1'); ?>
    </div>       
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'B2'); ?>
<?php echo $form->textField($model, 'B2'); ?>
        <?php echo $form->error($model, 'B2'); ?>
    </div>       
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'C025'); ?>
<?php echo $form->textField($model, 'C025'); ?>
        <?php echo $form->error($model, 'C025'); ?>
    </div>
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'C050'); ?>
<?php echo $form->textField($model, 'C050'); ?>
        <?php echo $form->error($model, 'C050'); ?>
    </div>
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'C1'); ?>
<?php echo $form->textField($model, 'C1'); ?>
        <?php echo $form->error($model, 'C1'); ?>
    </div>       
    <div class="forms33c">
        <?php echo $form->labelEx($model, 'C2'); ?>
<?php echo $form->textField($model, 'C2'); ?>
        <?php echo $form->error($model, 'C2'); ?>
    </div>       

    <div class="forms100c">
        <?php echo $form->labelEx($model, 'Observaciones'); ?>
<?php echo $form->textArea($model, 'Observaciones', array('styler' => 'width:300px;')); ?>
<?php echo $form->error($model, 'Observaciones'); ?>
    </div>
</div>
