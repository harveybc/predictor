<?php echo $form->errorSummary($model); ?>
<div class="forms100cb">
        <div class="row forms33c">
            <?php echo $form->labelEx($model, 'Fecha',array('styler' => '')); ?>
            <?php
            $today = date("Y-m-d");

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
                'htmlOptions' => array('style' => 'width:85%;'),
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
        
            <div class="row forms33c">
                <?php echo $form->labelEx($model, 'Orden de Trabajo',array('styler' => '')); ?>
                <?php echo $form->textField($model, 'OT',array('styler'=>'')); ?>
<?php echo $form->error($model, 'OT'); ?>
            </div>
                <div class="row forms33c">
                    <?php echo $form->labelEx($model, 'Estado',array('styler' => '')); ?>
                    <?php echo $form->dropDownList($model, 'Estado', array(
                            0=>'Adecuado',
                            1=>'Atención Requerida',
                            2=>'Malo',
                          ), array('styler' => ''));   
                    //textField($model, 'Estado', array('size' => 50, 'maxlength' => 50)); ?>
                    <?php echo $form->error($model, 'Estado'); ?>
                </div>
                <div class="row forms33c">
                    <?php
                    echo $form->labelEx($model, 'Buscar fotografía',array('styler' => ''));
                    echo $form->fileField($modelArchivo, 'nombre', array( 'style' => 'width:98%;height:22px;'));
                echo $form->error($modelArchivo, 'nombre');
                    ?> 

                </div>
                <div class="row forms33c">
                    <?php echo $form->labelEx($model, 'Analista',array('styler' => '')); ?>
<?php echo $form->textField($model, 'Analista', array('size' => 40, 'maxlength' => 50, 'styler'=>'')); ?> 

                </div> 
         


                <div class="row forms100c">
                    <?php echo $form->labelEx($model, 'Observaciones',array('styler' => '')); ?>
                    <?php echo $form->textArea($model, 'Observaciones', array('styler' => 'width:98%;')); ?>
                    <?php echo $form->error($model, 'Observaciones'); ?>
                </div>                    
</div>