<style>
    .forms33c,forms100c{
        text-align:left;
    }
    </style>
<div class="forms100cb">
            <div class="forms33c">
                <?php $today = date("Y-m-d"); ?>
                <?php echo $form->labelEx($model, 'Fecha', array('style' => 'text-align:left;')); ?>
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
                    //   'value' => $today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendarioCbm',
                    'htmlOptions' => array('style' => 'width:80%'),
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
                    <?php echo $form->labelEx($model, 'OT', array('style' => 'text-align:left;')); ?>
                    <?php echo $form->textField($model, 'OT', array('size' => 40, 'maxlength' => 50)); ?>
                    <?php echo $form->error($model, 'OT'); ?>
                </div>
                <div class="forms33c">
                    <?php echo $form->labelEx($model, 'Estado', array('style' => 'text-align:left;')); ?>
                    <?php
                    echo $form->dropDownList($model, 'Estado', array(
                        0 => 'Adecuado',
                        1 => 'Atención Requerida',
                        2 => 'Malo',
                            ), array('style' => 'width:98%;'));
                    //textField($model, 'Estado', array('size' => 50, 'maxlength' => 50)); 
                    ?>
<?php echo $form->error($model, 'Estado'); ?>
                </div>
                <div class="forms33c">
                    <?php echo $form->labelEx($model, 'Medicion', array('style' => 'text-align:left;')); ?>
                    <?php echo $form->textField($model, 'Medicion', array('size' => 40, 'maxlength' => 50)); ?>
                    <?php echo $form->error($model, 'Medicion'); ?>
                </div>
                <div class="forms33c">
                    <?php echo $form->labelEx($model, 'Tipo', array('style' => 'text-align:left;')); ?>
                    <?php echo $form->textField($model, 'Tipo', array('size' => 40, 'maxlength' => 50)); ?>
                    <?php echo $form->error($model, 'Tipo'); ?>
                </div>
                <div class="forms33c">
                    <?php echo $form->labelEx($model, 'Analista', array('style' => 'text-align:left;')); ?>
<?php echo $form->textField($model, 'Analista', array('size' => 40, 'maxlength' => 50)); ?>
<?php echo $form->error($model, 'Analista'); ?>
                </div>

    <div class="forms100c">
        <?php echo $form->labelEx($model, 'Observaciones', array('style' => 'padding-left:5px;text-align:left;')); ?>
<?php echo $form->textArea($model, 'Observaciones', array('styler' => 'width:93%;margin:0 4px 0 4px;')); ?>
<?php echo $form->error($model, 'Observaciones'); ?>
    </div>
</div>
