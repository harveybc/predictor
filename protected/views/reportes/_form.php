



<div class="forms100cb">

<?php echo $form->errorSummary($model); ?>
            <div class="row forms33c">
                <?php ?>

                <?php echo $form->labelEx($model, 'Fecha',array('styler' => 'text-align:left;')); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($model->Fecha))
                    $today = $model->Fecha;
                else
                    $model->Fecha = $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $model, //Model object
                    'attribute' => 'Fecha', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendarioCbm',
                    'htmlOptions'=>array('style'=>'width:85%'),
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
                <div class="row forms33c">
<?php echo $form->labelEx($model, 'Orden de Trabajo',array('styler' => 'text-align:left;')); ?>
                    <?php echo $form->textField($model, 'OT', array('size' => 60, 'maxlength' => 128, 'styler' => 'width:100%')); ?>
                    <?php echo $form->error($model, 'OT'); ?>
                </div>
                    <div class="row forms33c">
                    <?php echo $form->labelEx($model, 'Estado',array('styler' => 'text-align:left;')); ?>
                    <?php echo $form->dropDownList($model, 'Estado', array(
                            0=>'Adecuado',
                            1=>'Atención Requerida',
                            2=>'Malo',
                          ), array('styler' => 'width:350px;'));   
                    //textField($model, 'Estado', array('size' => 50, 'maxlength' => 50)); ?>
                    <?php echo $form->error($model, 'Estado'); ?> 
                </div>
            
                <div class="row forms33c">
<?php echo $form->labelEx($model, 'Analista',array('styler' => 'text-align:left;')); ?>
                    <?php echo $form->textField($model, 'Analista', array('size' => 50, 'maxlength' => 256, 'styler' => 'width:100%')); ?>
                    <?php echo $form->error($model, 'Analista'); ?>
                </div>    


                <div class="row forms33c">
<?php echo $form->labelEx($model, 'Gas',array('styler' => 'text-align:left;')); ?>

                
<?php
echo CHtml::activeDropDownList($model, 'Gas', array(
    'Aire' => "Aire",
    'CO2' => "CO2",
));
?>


<!--            <select name="Reportes[Gas]" id="Reportes_Gas">
    <option value="Aire">Aire</option>
    <option value="CO2">CO2</option>

</select>         ---> 

</div>

            
                <div class="row forms33c">
<?php echo $form->labelEx($model, 'Presion Gas Bar',array('styler' => 'text-align:left;')); ?>

                
                <!--
<select name="Reportes[Presion]" id="Reportes_Presion">
                        <option value="-">-</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="5">5</option>
                        <option value="7">7</option>
                        <option value="9">9</option>
                        <option value="10">10</option>

                </select>
                -->
<?php
echo CHtml::activeDropDownList($model, 'Presion', array(
    1 => "1",
    2 => "2",
    3 => "3",
    5 => "5",
    7 => "7",
    9 => "9",
    10 => "10",
));
?>
            </div>

           
                <div class="row forms33c">
<?php echo $form->labelEx($model, 'Nivel (db)',array('styler' => 'text-align:left;')); ?>

                
      <!--          <select name="Reportes[Decibeles]" id="Reportes_Decibeles">
                    <option value="-">-</option>
                    <option value="10">10</option>
                    <option value="20">20</option>
                    <option value="30">30</option>
                    <option value="40">40</option>
                    <option value="50">50</option>
                    <option value="60">60</option>
                    <option value="70">70</option>
                    <option value="80">80</option>
                    <option value="90">90</option>
                    <option value="100">100</option>

                </select>    --->

<?php
echo CHtml::activeDropDownList($model, 'Decibeles', array(
    10 => "10",
    20 => "20",
    30 => "30",
    40 => "40",
    50 => "50",
    60 => "60",
    70 => "70",
    80 => "80",
    90 => "90",
    100 => "100",
));
?>              


            </div>

        

                <div class="row forms33c">
<?php echo $form->labelEx($model, 'COSTO',array('styler' => 'text-align:left;')); ?>
                    <?php echo $form->textField($model, 'COSTO', array('styler' => 'width:100%')); ?>
                    <?php echo $form->error($model, 'COSTO'); ?>
                </div>


            
                <div class="row forms33c">
<?php echo $form->labelEx($model, 'CFM ',array('styler' => 'text-align:left;')); ?>
                    <?php echo $form->textField($model, 'CFM', array('styler' => 'width:100%')); ?>
                    <?php echo $form->error($model, 'CFM'); ?>
                </div>
            
            <div class="row forms100c">
<?php
echo $form->labelEx($model, 'Buscar fotografía',array('styler' => 'text-align:left;'));
                
                echo $form->fileField($modelArchivo, 'nombre', array( 'style' => 'width:98%;height:23px;'));
                echo $form->error($modelArchivo, 'nombre');
                ?>
            </div>
       

                        <div class="row forms100c">
<?php echo $form->labelEx($model, 'Descripción de la Fuga',array('styler' => 'text-align:left;')); ?>
                    <?php echo $form->textArea($model, 'Descripcion', array('rows' => 3,  'styler' => 'width:100%;')); ?>
                    <?php echo $form->error($model, 'Descripcion'); ?>
                </div>

            

       

</div>


