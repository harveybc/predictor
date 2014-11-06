<?php echo $form->errorSummary($model); ?>
<div class="forms100cb">
            <div class="forms33c">
                    <?php echo $form->labelEx($model, 'Orden de Trabajo'); ?>
                    <?php echo $form->textField($model, 'OT', array('styler' => 'width:137px;')); ?> 
                    <?php echo $form->error($model, 'OT'); ?>
<?php
if (isset($_GET['Toma'])) {
    $modeloAT = Aislamiento_tierra::model()->findByAttributes(array("Toma" => $_GET['Toma']));
    echo CHtml::hiddenField("Aislamiento_fases[Toma]", $_GET['Toma']);
    $model->Fecha=$modeloAT->Fecha;
    echo CHtml::hiddenField("Aislamiento_fases[TAG]", $modeloAT->TAG);
}
?>
            </div>
            <div class="forms33c">
                <?php $today = date("Y-m-d"); ?>
                <?php echo $form->labelEx($model, 'Fecha'); ?>
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
                     'htmlOptions' => array('style' => 'width:80%;text-align:left;'),
                    'options' => array(
                        'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                      'buttonImage'=>'/images/calendar.png',
                  'showOn'=> "both",
                  'buttonText'=>"Seleccione la fecha",
                  'buttonImageOnly'=> true 
                     
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
                    <?php echo $form->textArea($model, 'Observaciones', array()); ?>
                    <?php echo $form->error($model, 'Observaciones'); ?>
                </div>


</div>  






















