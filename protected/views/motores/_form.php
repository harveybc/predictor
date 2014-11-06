

<p class="note">Campos con <span class="required">*</span> son necesarios.</p>




<?php echo $form->errorSummary($model); ?>

<div styler="padding-left:10px;width:710px!important;margin-left:0px;margin-bottom:5px !important;color:#961C1F">

    
    <div>
        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'TAG'); ?>
                    <?php echo $form->textField($model, 'TAG', array('size' => 50, 'maxlength' => 50, 'style' => 'width:300px')); ?>
                    <?php echo $form->error($model, 'TAG'); ?>
                </div>
            </div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Codigo'); ?>
                    <?php echo $form->textField($model, 'Codigo', array('size' => 50, 'maxlength' => 50, 'style' => 'width:300px')); ?>
                    <?php echo $form->error($model, 'Codigo'); ?>
                </div>
            </div>
        </div>

        <div>

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Motor'); ?>
                    <?php echo $form->textField($model, 'Motor', array('size' => 60, 'maxlength' => 255, 'style' => 'width:300px')); ?>
                    <?php echo $form->error($model, 'Motor'); ?>
                </div>
            </div>

            <div>

                <div class="row">
                    <?php echo $form->labelEx($model, 'Marca'); ?>
                    <?php echo $form->textField($model, 'Marca', array('size' => 50, 'maxlength' => 50, 'style' => 'width:300px')); ?>
                    <?php echo $form->error($model, 'Marca'); ?>
                </div>

            </div>

        </div>

        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Modelo'); ?>
                    <?php echo $form->textField($model, 'Modelo', array('size' => 50, 'maxlength' => 50, 'style' => 'width:300px')); ?>
                    <?php echo $form->error($model, 'Modelo'); ?>
                </div>
            </div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Serie'); ?>
                    <?php echo $form->textField($model, 'Serie', array('size' => 50, 'maxlength' => 50, 'style' => 'width:300px')); ?>
                    <?php echo $form->error($model, 'Serie'); ?>
                </div>
            </div>
        </div>
    </div>
    <div styler="padding-left:10px;width:400px!important;margin-left:0px;margin-bottom:5px !important;color:#961C1F">    
        <div>
            <div class="row">

            </div>

        </div>
    </div>    
    <div styler="padding-left:10px;width:400px!important;margin-left:0px;margin-bottom:5px !important;color:#961C1F">    
        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'kW'); ?>
                    <?php echo $form->textField($model, 'kW', array('size' => 50, 'maxlength' => 50, 'style' => 'width:105px')); ?>
                    <?php echo $form->error($model, 'kW'); ?>
                </div>    
            </div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Velocidad'); ?>
                    <?php echo $form->textField($model, 'Velocidad', array('size' => 50, 'maxlength' => 50, 'style' => 'width:105px')); ?>
                    <?php echo $form->error($model, 'Velocidad'); ?>
                </div>
            </div>

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Rod_LC'); ?>
                    <?php echo $form->textField($model, 'Rod_LC', array('size' => 50, 'maxlength' => 50, 'style' => 'width:105px')); ?>
                    <?php echo $form->error($model, 'Rod_LC'); ?>
                </div>

            </div>

            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Rod_LA'); ?>
                    <?php echo $form->textField($model, 'Rod_LA', array('size' => 50, 'maxlength' => 50, 'style' => 'width:105px')); ?>
                    <?php echo $form->error($model, 'Rod_LA'); ?>
                </div>

            </div>

            <div>

                <div class="row">
                    <?php echo $form->labelEx($model, 'IP'); ?>
                    <?php echo $form->textField($model, 'IP', array('size' => 50, 'maxlength' => 50, 'style' => 'width:105px')); ?>
                    <?php echo $form->error($model, 'IP'); ?>
                </div>  

            </div>

         </div>
 
    </div>

    <div styler="padding-left:10px">
        
         <div>
            
            <div>

                <div class="row">
                    <?php echo $form->labelEx($model, 'Frame'); ?>
                    <?php echo $form->textField($model, 'Frame', array('size' => 50, 'maxlength' => 50, 'style' => 'width:105px')); ?>
                    <?php echo $form->error($model, 'Frame'); ?>
                </div>

            </div>

      
            <div styler="padding-left:15px">
                <div class="row">
                    <?php echo $form->labelEx($model, 'Lubricante'); ?>
                    <?php echo $form->textField($model, 'Lubricante', array('size' => 50, 'maxlength' => 50, 'style' => 'width:200px')); ?>
                    <?php echo $form->error($model, 'Lubricante'); ?>
                </div>

            </div>

            <div>  
<?php
echo $form->labelEx($model, 'Buscar fotografía',array('styler' => 'text-align:left;'));
                
                echo $form->fileField($modelArchivo, 'nombre', array( 'style' => 'width:98%;height:23px;'));
                echo $form->error($modelArchivo, 'nombre');
                ?>
    </div>  
        </div>


    </div>

</div>



