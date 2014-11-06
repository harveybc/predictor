<style type="text/css">

    .columna_h
    {
        width:6%; 
        text-align: center;
        margin-right: 0.9%;
        min-width: 25px;
    }
    .columna_h label{
        text-align: center;
    }
    
    .forms100c div{
        display: inline-block;
    }
</style>
<div class="forms100cb">
    <div class="row forms33c" styler="width:70%;">    
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
        ?>

        <!--      <div class="row">
<?php echo $form->labelEx($model, 'Fecha'); ?>
        <?php echo $form->textField($model, 'Fecha'); ?>
        <?php echo $form->error($model, 'Fecha'); ?>
              </div> --->
    </div>
    <div class="row forms33c">
        <?php echo $form->labelEx($model, 'Orden de Trabajo'); ?>
        <?php echo $form->textField($model, 'OT', array('style' => '')); ?> 
        <?php echo $form->error($model, 'OT'); ?>
    </div>

    <div class="row forms33c" >

<?php echo $form->labelEx($model, 'Estado'); ?>
        <?php
        echo $form->dropDownList($model, 'Estado', array(
            0 => 'Adecuado',
            1 => 'Atención Requerida',
            2 => 'Malo',
                ), array('style' => ''));
        //textField($model, 'Estado', array('size' => 50, 'maxlength' => 50)); 
        ?>
        <?php echo $form->error($model, 'Estado'); ?>
    </div>                                
    <div class="row">

        <?php echo $form->labelEx($model, 'Observaciones'); ?>
        <?php echo $form->textArea($model, 'Observaciones', array('style' => 'width:98%;')); ?>
        <?php echo $form->error($model, 'Observaciones'); ?>
    </div>
</div>
<div  class="forms100cb">
    Fase / Minutos
        <div class="forms100c">
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A', array('size' => 1, 'maxlength' => 128, 'style' => 'width:10px;')); ?>
            </div>  
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A025', array('style' => ';padding:0px;margin:0px;')); ?> 
                <?php echo $form->textField($model, 'A025', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A025'); ?>
            </div>
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A050', array('style' => ';padding:0px;margin:0px;')); ?>  
                <?php echo $form->textField($model, 'A050', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A050'); ?>
            </div>
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A1', array('style' => ';padding:0px;margin:0px;')); ?> 
                <?php echo $form->textField($model, 'A1', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A1'); ?>
            </div>
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A2', array('style' => ';padding:0px;margin:0px;')); ?>  
                <?php echo $form->textField($model, 'A2', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A2'); ?>
            </div>       
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A3', array('style' => ';padding:0px;margin:0px;')); ?>
                <?php echo $form->textField($model, 'A3', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A3'); ?>
            </div>
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A4', array('style' => ';padding:0px;margin:0px;')); ?> 
                <?php echo $form->textField($model, 'A4', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A4'); ?>
            </div>  
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A5', array('style' => ';padding:0px;margin:0px;')); ?>
                <?php echo $form->textField($model, 'A5', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A5'); ?>
            </div>   
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A6', array('style' => ';padding:0px;margin:0px;')); ?> 
                <?php echo $form->textField($model, 'A6', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A6'); ?>
            </div>
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A7', array('style' => ';padding:0px;margin:0px;')); ?> 
                <?php echo $form->textField($model, 'A7', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A7'); ?>
            </div>
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A8', array('style' => ';padding:0px;margin:0px;')); ?> 
                <?php echo $form->textField($model, 'A8', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A8'); ?>
            </div>
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A9', array('style' => ';padding:0px;margin:0px;')); ?> 
                <?php echo $form->textField($model, 'A9', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A9'); ?>
            </div>
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'A10', array('style' => ';padding:0px;margin:0px;')); ?> 
                <?php echo $form->textField($model, 'A10', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'A10'); ?>
            </div>
        </div>
        <div class="forms100c">
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'B', array('size' => 1, 'maxlength' => 128, 'style' => 'width:10px;')); ?>
            </div>
            <div class="columna_h">

                <?php echo $form->textField($model, 'B025', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B025'); ?>

            </div>
            <div class="columna_h">

                <?php echo $form->textField($model, 'B050', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B050'); ?>

            </div>
            <div class="columna_h">

                <?php echo $form->textField($model, 'B1', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B1'); ?>

            </div>


            <div class="columna_h">

                <?php echo $form->textField($model, 'B2', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B2'); ?>

            </div>

            <div class="columna_h">

                <?php echo $form->textField($model, 'B3', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B3'); ?>

            </div>

            <div class="columna_h">

                <?php echo $form->textField($model, 'B4', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B4'); ?>

            </div>

            <div class="columna_h">


                <?php echo $form->textField($model, 'B5', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B5'); ?>


            </div>


            <div class="columna_h">

                <?php echo $form->textField($model, 'B6', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B6'); ?>

            </div>



            <div class="columna_h">

                <?php echo $form->textField($model, 'B7', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B7'); ?>

            </div>


            <div class="columna_h">

                <?php echo $form->textField($model, 'B8', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B8'); ?>

            </div>


            <div class="columna_h">

                <?php echo $form->textField($model, 'B9', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B9'); ?>


            </div>

            <div class="columna_h">


                <?php echo $form->textField($model, 'B10', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'B10'); ?>

            </div>

        </div>         


        <div class="forms100c">
            <div class="columna_h">
                <?php echo $form->labelEx($model, 'C', array('size' => 1, 'maxlength' => 128, 'style' => 'width:10px;')); ?>

            </div>

            <div class="columna_h">

                <?php echo $form->textField($model, 'C025', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C025'); ?>

            </div>


            <div class="columna_h">


                <?php echo $form->textField($model, 'C050', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C050'); ?>

            </div>


            <div class="columna_h">

                <?php echo $form->textField($model, 'C1', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C1'); ?>

            </div>

            <div class="columna_h">

                <?php echo $form->textField($model, 'C2', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C2'); ?>

            </div>
            <div class="columna_h">

                <?php echo $form->textField($model, 'C3', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C3'); ?>

            </div>

            <div class="columna_h">

                <?php echo $form->textField($model, 'C4', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C4'); ?>

            </div>

            <div class="columna_h">

                <?php echo $form->textField($model, 'C5', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C5'); ?>

            </div>


            <div class="columna_h">

                <?php echo $form->textField($model, 'C6', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C6'); ?>

            </div>

            <div class="columna_h">

                <?php echo $form->textField($model, 'C7', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C7'); ?>

            </div>

            <div class="columna_h">

                <?php echo $form->textField($model, 'C8', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C8'); ?>

            </div>
            <div class="columna_h">

                <?php echo $form->textField($model, 'C9', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C9'); ?>

            </div>


            <div class="columna_h">

                <?php echo $form->textField($model, 'C10', array('size' => 1, 'maxlength' => 128, 'style' => '')); ?>
                <?php echo $form->error($model, 'C10'); ?>

            </div>

        </div>


    



</div>